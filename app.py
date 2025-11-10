from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageFilter, Image as PILImage
import numpy as np
import io, os, uuid, datetime, threading, time, logging, traceback, base64, re, gc, asyncio, psutil, uvicorn, cv2

# ---------- AI Model Load ----------
try:
    from rembg import remove as rembg_remove
    REMBG_AVAILABLE = True
    print("✅ Rembg AI model loaded successfully")
except ImportError:
    REMBG_AVAILABLE = False
    print("❌ Rembg not available, fallback mode active")

# ---------- Logging ----------
os.makedirs("logs", exist_ok=True)
logger = logging.getLogger("AI_API")
logger.setLevel(logging.INFO)
handler = logging.FileHandler("logs/app.log")
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)

# ---------- Folders ----------
for folder in ["uploads", "removebg", "vectorized", "static"]:
    os.makedirs(folder, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'webp', 'gif'}

# ---------- Memory Manager ----------
class MemoryManager:
    def __init__(self, interval=300):
        self.interval = interval
        self.running = False

    async def start(self):
        if not self.running:
            self.running = True
            asyncio.create_task(self._loop())
            logger.info(f"🚀 Memory Manager started ({self.interval}s interval)")

    async def _loop(self):
        while self.running:
            await asyncio.sleep(self.interval)
            await self.clean()

    async def clean(self):
        try:
            process = psutil.Process(os.getpid())
            before = process.memory_info().rss / 1024 / 1024
            collected = gc.collect()
            after = process.memory_info().rss / 1024 / 1024
            logger.info(f"🧹 Cleaned {collected} objs | Memory: {before:.1f}MB → {after:.1f}MB")
        except Exception as e:
            logger.error(f"Memory clean failed: {e}")

memory_manager = MemoryManager()

# ---------- FastAPI App ----------
app = FastAPI(title="AI Image API (Background Removal + SVG)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    await memory_manager.start()

# ---------- Utility ----------
def allowed_file(filename): return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ---------- Background Removal ----------
def fallback_background_remove(img: PILImage.Image) -> PILImage.Image:
    arr = np.array(img.convert('RGBA'))
    white_mask = (arr[:, :, 0] > 200) & (arr[:, :, 1] > 200) & (arr[:, :, 2] > 200)
    arr[:, :, 3] = np.where(white_mask, 0, 255)
    return PILImage.fromarray(arr, 'RGBA')

def crop_to_subject(img: PILImage.Image) -> PILImage.Image:
    if img.mode != 'RGBA': return img
    alpha = np.array(img)[:, :, 3]
    coords = np.argwhere(alpha > 0)
    if coords.size == 0: return img
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    return img.crop((x0, y0, x1 + 5, y1 + 5))

def refine_mask(img: PILImage.Image) -> PILImage.Image:
    arr = np.array(img)
    alpha = arr[:, :, 3]
    alpha = cv2.GaussianBlur(alpha, (5, 5), 0.5)
    arr[:, :, 3] = np.clip(alpha, 0, 255)
    return PILImage.fromarray(arr, 'RGBA')

def remove_background_optimized(image: PILImage.Image, quality: str):
    try:
        os.environ["U2NET_HOME"] = "/tmp/u2net"
        os.makedirs("/tmp/u2net", exist_ok=True)

        if REMBG_AVAILABLE:
            buf = io.BytesIO()
            image.convert("RGB").save(buf, format="PNG")
            out = rembg_remove(buf.getvalue(), post_process_mask=True)
            result = PILImage.open(io.BytesIO(out)).convert("RGBA")
        else:
            result = fallback_background_remove(image)

        if quality.lower() == "high":
            result = refine_mask(result)

        return crop_to_subject(result)
    except Exception as e:
        logger.error(f"remove_background_optimized error: {e}")
        return image.convert("RGBA")

# ---------- SVG Processor ----------
class SVGProcessor:
    async def colorful_svg(self, data, simplify=2, colors=32):
        img = PILImage.open(io.BytesIO(data)).convert('RGB')
        if 0 < colors < 256:
            img = img.quantize(colors=colors).convert('RGB')
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        w, h = img.size
        return f'''<svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg">
        <image href="data:image/png;base64,{b64}" width="100%" height="100%"/></svg>'''

svg_processor = SVGProcessor()

# ---------- ROUTES ----------
@app.get("/health")
async def health(): return {"status": "ok"}

@app.get("/")
async def root(): return FileResponse("static/index.html")

@app.post("/remove-bg")
async def remove_bg(image: UploadFile = File(...), background_color: str = Form("transparent"), quality: str = Form("high")):
    if not allowed_file(image.filename):
        return JSONResponse({"error": "File not allowed"}, 400)
    img = PILImage.open(io.BytesIO(await image.read())).convert("RGBA")
    out = remove_background_optimized(img, quality)

    bg_rgb = None
    fmt = "PNG"
    if background_color.startswith("#") and len(background_color) == 7:
        try:
            bg_rgb = tuple(int(background_color[i:i+2], 16) for i in (1, 3, 5))
        except: pass
    if bg_rgb:
        bg = PILImage.new("RGB", out.size, bg_rgb)
        bg.paste(out, mask=out.split()[3])
        out = bg.convert("RGB")
        fmt = "JPEG"

    file_id = f"nobg_{uuid.uuid4().hex}.{fmt.lower()}"
    path = os.path.join("removebg", file_id)
    out.save(path, format=fmt, quality=95)
    return {
        "success": True,
        "filename": file_id,
        "previewUrl": f"/get-image/{file_id}",
        "downloadUrl": f"/download/{file_id}",
        "ai_used": REMBG_AVAILABLE,
        "format": fmt
    }

@app.post("/api/remove-background")
async def alias_remove_background(image: UploadFile = File(...), background_color: str = Form("transparent"), quality: str = Form("high")):
    return await remove_bg(image=image, background_color=background_color, quality=quality)

@app.get("/get-image/{filename}")
async def get_image(filename: str):
    path = os.path.join("removebg", filename)
    if not os.path.exists(path): return JSONResponse({"error": "Not found"}, 404)
    media_type = "image/png" if filename.endswith(".png") else "image/jpeg"
    return FileResponse(path, media_type=media_type)

@app.get("/download/{filename}")
async def download(filename: str):
    path = os.path.join("removebg", filename)
    if not os.path.exists(path): return JSONResponse({"error": "Not found"}, 404)
    media_type = "image/png" if filename.endswith(".png") else "image/jpeg"
    return StreamingResponse(open(path, "rb"), media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename={filename}"})

@app.post("/vectorize")
async def vectorize(file: UploadFile = File(...)):
    img_data = await file.read()
    svg = await svg_processor.colorful_svg(img_data)
    return Response(content=svg, media_type="image/svg+xml")

# ---------- Cleanup Thread ----------
def cleanup_loop():
    while True:
        for folder in ["uploads", "removebg", "vectorized"]:
            for f in os.listdir(folder):
                p = os.path.join(folder, f)
                if os.path.isfile(p) and (time.time() - os.path.getctime(p)) > 3600:
                    os.remove(p)
        time.sleep(1800)
threading.Thread(target=cleanup_loop, daemon=True).start()

# ---------- Run ----------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 Server running on http://0.0.0.0:{port}")
    uvicorn.run("app:app", host="0.0.0.0", port=port)
