#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cat's diffusion bee 0.1 — A.C Holdings / Team Flames
DiffusionBee-style single-file FLUX studio (tkinter).

Engines
  FLUX.1  local  : black-forest-labs/FLUX.1-schnell | FLUX.1-dev   (diffusers FluxPipeline / FluxImg2ImgPipeline)
  FLUX.2  local  : black-forest-labs/FLUX.2-dev                    (diffusers Flux2Pipeline)
  FLUX.2  cloud  : flux-2-pro / flux-2-max / flux-2-flex / flux-2-klein via api.bfl.ai
  FLUX.3  cloud  : api.bfl.ai, slug editable (early-access; IDs not stable yet)

pip install --upgrade torch diffusers transformers accelerate sentencepiece protobuf pillow
(FLUX.1-dev / FLUX.2-dev are gated: `huggingface-cli login` after accepting the license.)
"""
import base64, io, json, os, queue, random, sys, threading, time, traceback, urllib.request
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

try:
    from PIL import Image, ImageTk
except ImportError:
    print("pip install pillow"); sys.exit(1)

APP = "cat's diffusion bee"
VER = "0.1"
HOME = os.path.join(os.path.expanduser("~"), "CatsDiffusionBee")
OUT = os.path.join(HOME, "outputs")
CFG = os.path.join(HOME, "config.json")
HIST = os.path.join(HOME, "history.json")
os.makedirs(OUT, exist_ok=True)

# ── theme: black buttons, blue-hue text ──────────────────────────────
BG      = "#05070d"
PANEL   = "#0b0f1a"
FIELD   = "#000000"
BTN     = "#000000"
BTN_HOV = "#0a1428"
BLUE    = "#4da3ff"
BLUE_HI = "#8cc8ff"
BLUE_LO = "#2a5c99"
BORDER  = "#16325c"
FONT    = ("Segoe UI", 10) if sys.platform == "win32" else ("Helvetica", 10)
FONT_B  = (FONT[0], 10, "bold")
FONT_H  = (FONT[0], 15, "bold")
MONO    = ("Consolas", 9) if sys.platform == "win32" else ("Courier", 9)

# ── model registry ───────────────────────────────────────────────────
MODELS = {
    "FLUX.1 [schnell] · local": dict(kind="local1", repo="black-forest-labs/FLUX.1-schnell",
                                    steps=4, guidance=0.0, max_seq=256),
    "FLUX.1 [dev] · local":     dict(kind="local1", repo="black-forest-labs/FLUX.1-dev",
                                    steps=28, guidance=3.5, max_seq=512),
    "FLUX.2 [dev] · local":     dict(kind="local2", repo="black-forest-labs/FLUX.2-dev",
                                    steps=28, guidance=4.0),
    "FLUX.2 [pro] · cloud":     dict(kind="api", slug="flux-2-pro",   steps=None, guidance=None),
    "FLUX.2 [max] · cloud":     dict(kind="api", slug="flux-2-max",   steps=None, guidance=None),
    "FLUX.2 [flex] · cloud":    dict(kind="api", slug="flux-2-flex",  steps=28,   guidance=4.5),
    "FLUX.2 [klein] · cloud":   dict(kind="api", slug="flux-2-klein", steps=None, guidance=None),
    "FLUX.3 · cloud":           dict(kind="api", slug="flux-3",       steps=None, guidance=None,
                                    editable_slug=True),
}
SIZES = ["512x512", "768x768", "1024x1024", "1024x768", "768x1024",
         "1152x896", "896x1152", "1344x768", "768x1344", "1536x640"]


def load_json(p, d):
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return d


def save_json(p, o):
    with open(p, "w", encoding="utf-8") as f:
        json.dump(o, f, indent=2)


# ── engines ──────────────────────────────────────────────────────────
class LocalEngine:
    """Lazy diffusers loader; caches one pipeline at a time to keep VRAM sane."""

    def __init__(self, log):
        self.log = log
        self.key = None
        self.pipe = None
        self.torch = None

    def device(self):
        t = self.torch
        if t.cuda.is_available():
            return "cuda"
        if getattr(t.backends, "mps", None) and t.backends.mps.is_available():
            return "mps"
        return "cpu"

    def load(self, spec, img2img, offload, hf_token):
        import torch
        self.torch = torch
        key = (spec["repo"], img2img, offload)
        if key == self.key and self.pipe is not None:
            return self.pipe
        self.unload()
        import diffusers
        if spec["kind"] == "local1":
            cls = diffusers.FluxImg2ImgPipeline if img2img else diffusers.FluxPipeline
        else:
            cls = getattr(diffusers, "Flux2Pipeline", None)
            if cls is None:
                raise RuntimeError("diffusers too old for FLUX.2 → pip install -U diffusers")
        dev = self.device()
        dtype = torch.bfloat16 if dev in ("cuda", "cpu") else torch.float16
        self.log(f"loading {spec['repo']} ({cls.__name__}, {dev}, {dtype})…")
        kw = dict(torch_dtype=dtype)
        if hf_token:
            kw["token"] = hf_token
        pipe = cls.from_pretrained(spec["repo"], **kw)
        if offload and dev == "cuda":
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(dev)
        try:
            pipe.vae.enable_tiling()
        except Exception:
            pass
        self.pipe, self.key = pipe, key
        self.log("model ready.")
        return pipe

    def unload(self):
        if self.pipe is not None:
            del self.pipe
            self.pipe, self.key = None, None
            try:
                import gc; gc.collect()
                if self.torch and self.torch.cuda.is_available():
                    self.torch.cuda.empty_cache()
            except Exception:
                pass

    def run(self, spec, p, progress, cancel):
        img2img = p["init_image"] is not None
        pipe = self.load(spec, img2img and spec["kind"] == "local1", p["offload"], p["hf_token"])
        torch = self.torch
        total = p["steps"]

        def cb(_pipe, step, _t, kwargs):
            if cancel.is_set():
                _pipe._interrupt = True
            progress(step + 1, total)
            return kwargs

        out = []
        for i in range(p["count"]):
            if cancel.is_set():
                break
            seed = p["seed"] + i
            gen = torch.Generator("cpu").manual_seed(seed)
            kw = dict(prompt=p["prompt"], num_inference_steps=total, guidance_scale=p["guidance"],
                      generator=gen, callback_on_step_end=cb)
            if spec["kind"] == "local1":
                kw["max_sequence_length"] = spec.get("max_seq", 512)
                if img2img:
                    kw["image"] = p["init_image"].resize((p["w"], p["h"]))
                    kw["strength"] = p["strength"]
                kw["height"], kw["width"] = p["h"], p["w"]
            else:
                kw["height"], kw["width"] = p["h"], p["w"]
                if img2img:
                    kw["image"] = [p["init_image"]]
            img = pipe(**kw).images[0]
            out.append((img, seed))
        return out


class BFLEngine:
    """api.bfl.ai: POST /v1/{slug} → poll polling_url → download result.sample."""

    def __init__(self, log):
        self.log = log

    def _req(self, url, key, data=None):
        h = {"accept": "application/json", "x-key": key}
        body = None
        if data is not None:
            h["Content-Type"] = "application/json"
            body = json.dumps(data).encode()
        r = urllib.request.Request(url, data=body, headers=h, method="POST" if data is not None else "GET")
        with urllib.request.urlopen(r, timeout=60) as resp:
            return json.loads(resp.read().decode())

    def run(self, spec, p, progress, cancel):
        key = p["bfl_key"] or os.environ.get("BFL_API_KEY", "")
        if not key:
            raise RuntimeError("no BFL API key (Settings tab or $BFL_API_KEY)")
        slug = p["slug"] or spec["slug"]
        base = p["bfl_base"].rstrip("/")
        out = []
        for i in range(p["count"]):
            if cancel.is_set():
                break
            seed = p["seed"] + i
            body = dict(prompt=p["prompt"], width=p["w"], height=p["h"], seed=seed, output_format="png")
            if spec.get("steps"):
                body["steps"] = p["steps"]
                body["guidance"] = p["guidance"]
            if p["init_image"] is not None:
                buf = io.BytesIO(); p["init_image"].save(buf, "PNG")
                body["input_image"] = base64.b64encode(buf.getvalue()).decode()
            self.log(f"POST {base}/v1/{slug}")
            job = self._req(f"{base}/v1/{slug}", key, body)
            poll = job.get("polling_url") or f"{base}/v1/get_result?id={job['id']}"
            t0, tick = time.time(), 0
            while True:
                if cancel.is_set():
                    return out
                time.sleep(0.6)
                tick += 1
                progress(min(tick, 95), 100)
                st = self._req(poll, key)
                s = st.get("status")
                if s == "Ready":
                    url = st["result"]["sample"]
                    with urllib.request.urlopen(url, timeout=120) as r:
                        img = Image.open(io.BytesIO(r.read())).convert("RGB")
                    out.append((img, seed))
                    progress(100, 100)
                    break
                if s in ("Error", "Failed", "Request Moderated", "Content Moderated", "Task not found"):
                    raise RuntimeError(f"BFL: {s} {st.get('details', '')}")
                if time.time() - t0 > 600:
                    raise RuntimeError("BFL: timeout")
        return out


# ── widgets ──────────────────────────────────────────────────────────
class BeeButton(tk.Label):
    def __init__(self, master, text, cmd, big=False, **kw):
        super().__init__(master, text=text, bg=BTN, fg=BLUE, cursor="hand2",
                         font=(FONT[0], 12 if big else 10, "bold"),
                         padx=18 if big else 12, pady=10 if big else 5,
                         highlightthickness=1, highlightbackground=BORDER, **kw)
        self.cmd, self.enabled = cmd, True
        self.bind("<Enter>", lambda e: self.enabled and self.config(bg=BTN_HOV, fg=BLUE_HI))
        self.bind("<Leave>", lambda e: self.config(bg=BTN, fg=BLUE if self.enabled else BLUE_LO))
        self.bind("<Button-1>", lambda e: self.enabled and self.cmd())

    def set_enabled(self, on):
        self.enabled = on
        self.config(fg=BLUE if on else BLUE_LO, cursor="hand2" if on else "arrow")


def lbl(m, t, **kw):
    return tk.Label(m, text=t, bg=kw.pop("bg", PANEL), fg=kw.pop("fg", BLUE), font=kw.pop("font", FONT), **kw)


def entry(m, var, w=10, show=None):
    return tk.Entry(m, textvariable=var, width=w, bg=FIELD, fg=BLUE_HI, insertbackground=BLUE,
                    relief="flat", highlightthickness=1, highlightbackground=BORDER,
                    highlightcolor=BLUE, font=FONT, show=show)


# ── app ──────────────────────────────────────────────────────────────
class CatsDiffusionBee(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"{APP} {VER}")
        self.geometry("1280x820")
        self.minsize(1000, 680)
        self.configure(bg=BG)
        self.cfg = load_json(CFG, {"bfl_key": "", "hf_token": "", "bfl_base": "https://api.bfl.ai",
                                   "offload": True, "flux3_slug": "flux-3"})
        self.history = load_json(HIST, [])
        self.q = queue.Queue()
        self.cancel = threading.Event()
        self.busy = False
        self.init_image = None
        self.results = []
        self.thumbs = []
        self.local = LocalEngine(self.log)
        self.api = BFLEngine(self.log)
        self._style()
        self._build()
        self.after(50, self._pump)
        self.protocol("WM_DELETE_WINDOW", self._quit)

    # styling for ttk bits (combobox, scale, progress)
    def _style(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure("TCombobox", fieldbackground=FIELD, background=BTN, foreground=BLUE_HI,
                    arrowcolor=BLUE, bordercolor=BORDER, lightcolor=FIELD, darkcolor=FIELD)
        s.map("TCombobox", fieldbackground=[("readonly", FIELD)], foreground=[("readonly", BLUE_HI)],
              selectbackground=[("readonly", FIELD)], selectforeground=[("readonly", BLUE_HI)])
        self.option_add("*TCombobox*Listbox.background", FIELD)
        self.option_add("*TCombobox*Listbox.foreground", BLUE_HI)
        self.option_add("*TCombobox*Listbox.selectBackground", BTN_HOV)
        self.option_add("*TCombobox*Listbox.selectForeground", BLUE_HI)
        s.configure("Bee.Horizontal.TProgressbar", troughcolor=FIELD, background=BLUE,
                    bordercolor=BORDER, lightcolor=BLUE, darkcolor=BLUE_LO)
        s.configure("Bee.Horizontal.TScale", background=PANEL, troughcolor=FIELD)

    def _build(self):
        # sidebar
        side = tk.Frame(self, bg=PANEL, width=190, highlightthickness=1, highlightbackground=BORDER)
        side.pack(side="left", fill="y")
        side.pack_propagate(False)
        lbl(side, "🐝 cat's", font=FONT_H).pack(pady=(18, 0))
        lbl(side, "diffusion bee", font=FONT_H).pack()
        lbl(side, f"v{VER} · FLUX 1 · 2 · 3", fg=BLUE_LO).pack(pady=(0, 18))
        self.pages = {}
        self.navbtn = {}
        body = tk.Frame(self, bg=BG)
        body.pack(side="left", fill="both", expand=True)
        for name in ("Text to Image", "Image to Image", "History", "Settings"):
            b = BeeButton(side, name, lambda n=name: self.show(n))
            b.pack(fill="x", padx=12, pady=4)
            self.navbtn[name] = b
            self.pages[name] = tk.Frame(body, bg=BG)
        self.status = lbl(side, "idle", fg=BLUE_LO, wraplength=170, justify="left")
        self.status.pack(side="bottom", pady=12, padx=10)

        self._gen_page(self.pages["Text to Image"], img2img=False)
        self._gen_page(self.pages["Image to Image"], img2img=True)
        self._history_page(self.pages["History"])
        self._settings_page(self.pages["Settings"])
        self.show("Text to Image")

    def show(self, name):
        for n, f in self.pages.items():
            f.pack_forget()
            self.navbtn[n].config(highlightbackground=BLUE if n == name else BORDER)
        self.pages[name].pack(fill="both", expand=True)
        self.current = name
        if name == "History":
            self._refresh_history()

    # ── generate page (shared by t2i / i2i) ──
    def _gen_page(self, page, img2img):
        v = dict(model=tk.StringVar(value=list(MODELS)[0]), size=tk.StringVar(value="1024x1024"),
                 steps=tk.IntVar(value=4), guidance=tk.DoubleVar(value=0.0), seed=tk.StringVar(value="-1"),
                 count=tk.IntVar(value=1), strength=tk.DoubleVar(value=0.75))
        top = tk.Frame(page, bg=BG)
        top.pack(fill="x", padx=16, pady=(14, 6))
        prompt = tk.Text(top, height=4, bg=FIELD, fg=BLUE_HI, insertbackground=BLUE, wrap="word",
                         relief="flat", highlightthickness=1, highlightbackground=BORDER,
                         highlightcolor=BLUE, font=(FONT[0], 11))
        prompt.pack(side="left", fill="x", expand=True)
        prompt.insert("1.0", "a cozy pixel-art cat hacker at a CRT, neon blue glow, 16-bit")
        prompt.bind("<Control-Return>", lambda e: (self.generate(page), "break")[1])
        gen = BeeButton(top, "Generate", lambda: self.generate(page), big=True)
        gen.pack(side="left", padx=(10, 0), fill="y")

        opts = tk.Frame(page, bg=PANEL, highlightthickness=1, highlightbackground=BORDER)
        opts.pack(fill="x", padx=16, pady=4)
        r = 0

        def row(label, w, col):
            lbl(opts, label).grid(row=r, column=col, sticky="e", padx=(10, 4), pady=6)
            w.grid(row=r, column=col + 1, sticky="w", padx=(0, 10), pady=6)

        cb = ttk.Combobox(opts, textvariable=v["model"], values=list(MODELS), state="readonly", width=26)
        row("Model", cb, 0)
        row("Size", ttk.Combobox(opts, textvariable=v["size"], values=SIZES, width=11), 2)
        row("Images", tk.Spinbox(opts, from_=1, to=8, textvariable=v["count"], width=4, bg=FIELD, fg=BLUE_HI,
                                 buttonbackground=BTN, relief="flat", insertbackground=BLUE), 4)
        r = 1
        stp = tk.Spinbox(opts, from_=1, to=100, textvariable=v["steps"], width=5, bg=FIELD, fg=BLUE_HI,
                         buttonbackground=BTN, relief="flat", insertbackground=BLUE)
        row("Steps", stp, 0)
        gui = tk.Spinbox(opts, from_=0.0, to=30.0, increment=0.5, textvariable=v["guidance"], width=5,
                         bg=FIELD, fg=BLUE_HI, buttonbackground=BTN, relief="flat", insertbackground=BLUE)
        row("Guidance", gui, 2)
        sf = tk.Frame(opts, bg=PANEL)
        entry(sf, v["seed"], 12).pack(side="left")
        BeeButton(sf, "🎲", lambda: v["seed"].set(str(random.randint(0, 2**32 - 1)))).pack(side="left", padx=4)
        row("Seed", sf, 4)
        if img2img:
            r = 2
            sc = tk.Scale(opts, from_=0.05, to=1.0, resolution=0.05, orient="horizontal", variable=v["strength"],
                          bg=PANEL, fg=BLUE, troughcolor=FIELD, highlightthickness=0, length=180,
                          activebackground=BLUE)
            row("Strength", sc, 0)
            pick = tk.Frame(opts, bg=PANEL)
            BeeButton(pick, "Load input image…", lambda: self._pick_init(page)).pack(side="left")
            page.init_lbl = lbl(pick, "none", fg=BLUE_LO)
            page.init_lbl.pack(side="left", padx=8)
            row("Input", pick, 2)

        def on_model(*_):
            s = MODELS[v["model"].get()]
            if s.get("steps") is not None:
                v["steps"].set(s["steps"]); stp.config(state="normal")
            else:
                stp.config(state="disabled")
            if s.get("guidance") is not None:
                v["guidance"].set(s["guidance"]); gui.config(state="normal")
            else:
                gui.config(state="disabled")
        cb.bind("<<ComboboxSelected>>", on_model)

        prog = tk.Frame(page, bg=BG)
        prog.pack(fill="x", padx=16, pady=(6, 0))
        pb = ttk.Progressbar(prog, style="Bee.Horizontal.TProgressbar", maximum=100)
        pb.pack(side="left", fill="x", expand=True)
        stop = BeeButton(prog, "Stop", self.cancel.set)
        stop.pack(side="left", padx=(10, 0))

        # canvas / gallery
        view = tk.Frame(page, bg=BG)
        view.pack(fill="both", expand=True, padx=16, pady=10)
        canvas = tk.Canvas(view, bg=FIELD, highlightthickness=1, highlightbackground=BORDER)
        canvas.pack(side="left", fill="both", expand=True)
        strip = tk.Frame(view, bg=PANEL, width=150, highlightthickness=1, highlightbackground=BORDER)
        strip.pack(side="left", fill="y", padx=(10, 0))
        strip.pack_propagate(False)
        BeeButton(strip, "Save…", lambda: self._save_current(page)).pack(fill="x", padx=8, pady=(8, 4))
        BeeButton(strip, "→ Img2Img", lambda: self._send_i2i(page)).pack(fill="x", padx=8, pady=4)
        thumbs = tk.Frame(strip, bg=PANEL)
        thumbs.pack(fill="both", expand=True, pady=6)
        canvas.bind("<Configure>", lambda e: self._draw(page))

        logw = tk.Text(page, height=4, bg=FIELD, fg=BLUE_LO, font=MONO, relief="flat",
                       highlightthickness=1, highlightbackground=BORDER, state="disabled")
        logw.pack(fill="x", padx=16, pady=(0, 12))

        page.v, page.prompt, page.gen, page.pb, page.canvas = v, prompt, gen, pb, canvas
        page.thumbs, page.logw, page.img2img = thumbs, logw, img2img
        page.shown, page.tk_img, page.init = None, None, None
        page.flux3_slug = None

    def _pick_init(self, page):
        p = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp")])
        if p:
            page.init = Image.open(p).convert("RGB")
            page.init_lbl.config(text=f"{os.path.basename(p)}  {page.init.size[0]}x{page.init.size[1]}", fg=BLUE)

    def _send_i2i(self, page):
        if page.shown is None:
            return
        tgt = self.pages["Image to Image"]
        tgt.init = page.shown.copy()
        tgt.init_lbl.config(text=f"from gallery  {tgt.init.size[0]}x{tgt.init.size[1]}", fg=BLUE)
        tgt.prompt.delete("1.0", "end"); tgt.prompt.insert("1.0", page.prompt.get("1.0", "end").strip())
        self.show("Image to Image")

    def _draw(self, page):
        c = page.canvas
        c.delete("all")
        cw, ch = max(c.winfo_width(), 2), max(c.winfo_height(), 2)
        if page.shown is None:
            c.create_text(cw // 2, ch // 2, text="🐝  your images land here", fill=BLUE_LO, font=FONT_H)
            return
        im = page.shown.copy()
        im.thumbnail((cw - 8, ch - 8), Image.LANCZOS)
        page.tk_img = ImageTk.PhotoImage(im)
        c.create_image(cw // 2, ch // 2, image=page.tk_img)

    def _add_thumb(self, page, img, meta):
        t = img.copy(); t.thumbnail((128, 128))
        ti = ImageTk.PhotoImage(t)
        self.thumbs.append(ti)
        b = tk.Label(page.thumbs, image=ti, bg=PANEL, cursor="hand2",
                     highlightthickness=1, highlightbackground=BORDER)
        b.pack(pady=3)
        b.bind("<Button-1>", lambda e: (setattr(page, "shown", img), self._draw(page)))
        kids = page.thumbs.winfo_children()
        for k in kids[:-6]:
            k.destroy()

    def _save_current(self, page):
        if page.shown is None:
            return
        p = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
        if p:
            page.shown.save(p)
            self.log(f"saved {p}")

    # ── generation ──
    def generate(self, page):
        if self.busy:
            return
        v = page.v
        prompt = page.prompt.get("1.0", "end").strip()
        if not prompt:
            return
        name = v["model"].get()
        spec = MODELS[name]
        try:
            w, h = (int(x) for x in v["size"].get().lower().split("x"))
            w, h = w // 16 * 16, h // 16 * 16
        except Exception:
            messagebox.showerror(APP, "size must be WxH"); return
        if page.img2img and page.init is None:
            messagebox.showerror(APP, "load an input image first"); return
        try:
            seed = int(v["seed"].get())
        except ValueError:
            seed = -1
        if seed < 0:
            seed = random.randint(0, 2**32 - 1)
        params = dict(prompt=prompt, w=w, h=h, steps=int(v["steps"].get()), guidance=float(v["guidance"].get()),
                      seed=seed, count=int(v["count"].get()), strength=float(v["strength"].get()),
                      init_image=page.init if page.img2img else None,
                      offload=self.cfg.get("offload", True), hf_token=self.cfg.get("hf_token", ""),
                      bfl_key=self.cfg.get("bfl_key", ""), bfl_base=self.cfg.get("bfl_base", "https://api.bfl.ai"),
                      slug=self.cfg.get("flux3_slug", "flux-3") if spec.get("editable_slug") else None)
        self.busy = True
        self.cancel.clear()
        self._log_page = page
        page.gen.set_enabled(False)
        page.pb["value"] = 0
        self._set_status(f"generating · {name}")
        threading.Thread(target=self._worker, args=(page, name, spec, params), daemon=True).start()

    def _worker(self, page, name, spec, p):
        t0 = time.time()
        try:
            eng = self.api if spec["kind"] == "api" else self.local
            prog = lambda a, b: self.q.put(("prog", page, a / max(b, 1) * 100))
            res = eng.run(spec, p, prog, self.cancel)
            for img, seed in res:
                fn = os.path.join(OUT, f"cdb_{int(time.time()*1000)}_{seed}.png")
                img.save(fn)
                meta = dict(file=fn, prompt=p["prompt"], model=name, seed=seed, w=p["w"], h=p["h"],
                            steps=p["steps"], guidance=p["guidance"], time=time.strftime("%Y-%m-%d %H:%M:%S"),
                            img2img=p["init_image"] is not None)
                self.q.put(("img", page, img, meta))
            self.q.put(("done", page, f"done · {len(res)} img · {time.time()-t0:.1f}s"))
        except Exception as e:
            self.q.put(("log", traceback.format_exc()))
            self.q.put(("done", page, f"error: {e}"))

    def _pump(self):
        try:
            while True:
                m = self.q.get_nowait()
                k = m[0]
                if k == "prog":
                    m[1].pb["value"] = m[2]
                elif k == "img":
                    _, page, img, meta = m
                    page.shown = img
                    self._draw(page)
                    self._add_thumb(page, img, meta)
                    self.history.insert(0, meta)
                    save_json(HIST, self.history[:500])
                elif k == "log":
                    self._log_now(m[1])
                elif k == "done":
                    _, page, msg = m
                    self.busy = False
                    page.gen.set_enabled(True)
                    page.pb["value"] = 100 if not msg.startswith("error") else 0
                    self._set_status(msg)
                    self._log_now(msg)
        except queue.Empty:
            pass
        self.after(50, self._pump)

    def log(self, s):
        self.q.put(("log", s))

    def _log_now(self, s):
        for name in ("Text to Image", "Image to Image"):
            w = self.pages[name].logw
            w.config(state="normal")
            w.insert("end", s.rstrip() + "\n")
            w.see("end")
            w.config(state="disabled")

    def _set_status(self, s):
        self.status.config(text=s)

    # ── history ──
    def _history_page(self, page):
        bar = tk.Frame(page, bg=BG)
        bar.pack(fill="x", padx=16, pady=14)
        lbl(bar, "History", font=FONT_H, bg=BG).pack(side="left")
        BeeButton(bar, "Open folder", self._open_out).pack(side="right")
        BeeButton(bar, "Clear", self._clear_hist).pack(side="right", padx=8)
        wrap = tk.Frame(page, bg=BG)
        wrap.pack(fill="both", expand=True, padx=16, pady=(0, 14))
        c = tk.Canvas(wrap, bg=BG, highlightthickness=0)
        sb = tk.Scrollbar(wrap, orient="vertical", command=c.yview, bg=BTN, troughcolor=FIELD)
        c.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        c.pack(side="left", fill="both", expand=True)
        inner = tk.Frame(c, bg=BG)
        c.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda e: c.configure(scrollregion=c.bbox("all")))
        c.bind_all("<MouseWheel>", lambda e: self.current == "History" and c.yview_scroll(-e.delta // 120, "units"))
        page.inner, page.hist_imgs = inner, []

    def _refresh_history(self):
        page = self.pages["History"]
        for w in page.inner.winfo_children():
            w.destroy()
        page.hist_imgs.clear()
        cols = 5
        for i, h in enumerate(self.history[:100]):
            if not os.path.exists(h["file"]):
                continue
            card = tk.Frame(page.inner, bg=PANEL, highlightthickness=1, highlightbackground=BORDER)
            card.grid(row=i // cols, column=i % cols, padx=6, pady=6, sticky="n")
            try:
                im = Image.open(h["file"]); im.thumbnail((180, 180))
                ti = ImageTk.PhotoImage(im); page.hist_imgs.append(ti)
                il = tk.Label(card, image=ti, bg=PANEL, cursor="hand2")
                il.pack(padx=4, pady=4)
                il.bind("<Button-1>", lambda e, hh=h: self._reuse(hh))
            except Exception:
                continue
            lbl(card, h["prompt"][:60], wraplength=180, justify="left").pack(padx=4)
            lbl(card, f"{h['model']}\nseed {h['seed']} · {h['w']}x{h['h']}", fg=BLUE_LO,
                font=(FONT[0], 8), justify="left").pack(padx=4, pady=(0, 4))

    def _reuse(self, h):
        page = self.pages["Image to Image" if h.get("img2img") else "Text to Image"]
        page.prompt.delete("1.0", "end"); page.prompt.insert("1.0", h["prompt"])
        if h["model"] in MODELS:
            page.v["model"].set(h["model"])
        page.v["seed"].set(str(h["seed"])); page.v["size"].set(f"{h['w']}x{h['h']}")
        page.v["steps"].set(h["steps"]); page.v["guidance"].set(h["guidance"])
        page.shown = Image.open(h["file"]).convert("RGB")
        self.show("Image to Image" if h.get("img2img") else "Text to Image")
        self._draw(page)

    def _clear_hist(self):
        if messagebox.askyesno(APP, "clear history list? (files stay in outputs/)"):
            self.history = []; save_json(HIST, []); self._refresh_history()

    def _open_out(self):
        if sys.platform == "win32":
            os.startfile(OUT)
        elif sys.platform == "darwin":
            os.system(f'open "{OUT}"')
        else:
            os.system(f'xdg-open "{OUT}" &')

    # ── settings ──
    def _settings_page(self, page):
        lbl(page, "Settings", font=FONT_H, bg=BG).pack(anchor="w", padx=16, pady=14)
        box = tk.Frame(page, bg=PANEL, highlightthickness=1, highlightbackground=BORDER)
        box.pack(fill="x", padx=16)
        sv = dict(bfl_key=tk.StringVar(value=self.cfg.get("bfl_key", "")),
                  hf_token=tk.StringVar(value=self.cfg.get("hf_token", "")),
                  bfl_base=tk.StringVar(value=self.cfg.get("bfl_base", "https://api.bfl.ai")),
                  flux3_slug=tk.StringVar(value=self.cfg.get("flux3_slug", "flux-3")),
                  offload=tk.BooleanVar(value=self.cfg.get("offload", True)))
        rows = [("BFL API key", "bfl_key", "*"), ("HF token (gated repos)", "hf_token", "*"),
                ("BFL base URL", "bfl_base", None), ("FLUX.3 endpoint slug", "flux3_slug", None)]
        for i, (t, k, show) in enumerate(rows):
            lbl(box, t).grid(row=i, column=0, sticky="e", padx=10, pady=6)
            entry(box, sv[k], 48, show).grid(row=i, column=1, sticky="w", padx=10, pady=6)
        tk.Checkbutton(box, text="CPU offload (low VRAM)", variable=sv["offload"], bg=PANEL, fg=BLUE,
                       selectcolor=FIELD, activebackground=PANEL, activeforeground=BLUE_HI,
                       font=FONT).grid(row=len(rows), column=1, sticky="w", padx=10, pady=6)
        bar = tk.Frame(page, bg=BG)
        bar.pack(fill="x", padx=16, pady=10)

        def save():
            for k, var in sv.items():
                self.cfg[k] = var.get()
            save_json(CFG, self.cfg)
            self._set_status("settings saved")
        BeeButton(bar, "Save", save).pack(side="left")
        BeeButton(bar, "Unload local model", lambda: (self.local.unload(), self._set_status("model unloaded"))
                  ).pack(side="left", padx=8)
        info = ("Local: FLUX.1 schnell (Apache-2.0, 4 steps), FLUX.1 dev, FLUX.2 dev — via diffusers, CUDA/MPS/CPU.\n"
                "Cloud: FLUX.2 pro/max/flex/klein + FLUX.3 via api.bfl.ai (x-key header, polled).\n"
                f"Outputs: {OUT}")
        lbl(page, info, bg=BG, fg=BLUE_LO, justify="left").pack(anchor="w", padx=16, pady=8)

    def _quit(self):
        self.cancel.set()
        self.destroy()


if __name__ == "__main__":
    CatsDiffusionBee().mainloop()
