#!/usr/bin/env python3
"""Image-count ladder with VRAM-peak sampling: find the real per-request image ceiling.

Each step sends one non-streaming chat request containing N identical noise PNGs
(default 2 Mpx area, the server's mm-process-config cap) while a background thread
samples `nvidia-smi` every 0.2 s. Records wall time, GPU peak, and pass/fail. Stops
on failure or when the server dies.

What this proved on the target card (2026-09-11): VRAM never binds - GPU peak is flat
from 1 to 96 images. The ceiling is the context window: each 2 Mpx image costs ~2050
tokens, so ~127 images exhaust 262144 and the server answers 400. See evidence/README.md.

Env: SGLANG_BASE (default http://127.0.0.1:8080), MODEL_NAME (default qwen3.8-27b)
"""
import argparse, base64, json, os, random, struct, subprocess, threading, time, urllib.request, zlib

BASE = os.environ.get("SGLANG_BASE", "http://127.0.0.1:8080").rstrip("/")
MODEL = os.environ.get("MODEL_NAME", "qwen3.8-27b")

ap = argparse.ArgumentParser()
ap.add_argument("--sizes", default="1,4,8,16,24,32,40,64,96,128")
ap.add_argument("--px", type=int, default=2097152, help="target image area in pixels")
ap.add_argument("--w", type=int, default=2048)
args = ap.parse_args()


def png_bytes(w, h, seed=7):
    """Random-noise RGB PNG, pure stdlib (compress level 0 => size ~ w*h*3)."""
    rnd = random.Random(seed)
    def chunk(tag, payload):
        c = struct.pack(">I", len(payload)) + tag + payload
        return c + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
    raw = b"".join(b"\x00" + bytes(rnd.randrange(256) for _ in range(w * 3)) for _ in range(h))
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 0)) + chunk(b"IEND", b""))


def make_url(w, h):
    try:
        from PIL import Image
        img = Image.new("RGB", (w, h))
        px = img.load()
        rnd = random.Random(7)
        for y in range(h):
            for x in range(w):
                px[x, y] = (rnd.randrange(256), rnd.randrange(256), rnd.randrange(256))
        import io
        buf = io.BytesIO(); img.save(buf, "PNG", compress_level=0)
        data = buf.getvalue()
    except ImportError:
        data = png_bytes(w, h)
    return "data:image/png;base64," + base64.b64encode(data).decode()


mem_used = [0]
stop = threading.Event()


def sampler():
    while not stop.is_set():
        try:
            out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used",
                                           "--format=csv,noheader,nounits"]).decode().strip()
            mem_used[0] = max(mem_used[0], int(out.splitlines()[0]))
        except Exception:
            pass
        time.sleep(0.2)


w = args.w
h = args.px // w
print(f"image {w}x{h} = {w*h/1e6:.2f} Mpx")
u = make_url(w, h)
print(f"base64 len {len(u)/1e6:.1f} MB")

for n in [int(x) for x in args.sizes.split(",")]:
    content = [{"type": "image_url", "image_url": {"url": u}} for _ in range(n)]
    content.append({"type": "text", "text": "Name one common feature of these images."})
    body = json.dumps({"model": MODEL,
                       "messages": [{"role": "user", "content": content}],
                       "max_tokens": 64}).encode()
    mem_used[0] = 0
    stop.clear()
    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    t0 = time.time()
    status, err = "ok", ""
    try:
        r = urllib.request.Request(BASE + "/v1/chat/completions", data=body,
                                   headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(r, timeout=900) as f:
            json.load(f)
    except Exception as e:
        status, err = "fail", str(e)[:150]
        try:
            err += " :: " + e.read().decode()[:200]  # HTTPError body carries the reason
        except Exception:
            pass
    wall = time.time() - t0
    stop.set()
    th.join()
    alive = ""
    try:
        subprocess.check_call(["curl", "-sf", "-m", "10", BASE + "/health"],
                              stdout=subprocess.DEVNULL)
    except Exception:
        alive = " SERVER-DOWN"
    print(f"N={n:3d}  {status}  wall={wall:6.1f}s  gpu_peak={mem_used[0]}MiB{alive}  {err}", flush=True)
    if alive or status == "fail":
        break
