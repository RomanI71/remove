from fastapi.responses import JSONResponse, FileResponse, Response
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles 
from fastapi.responses import FileResponse
from fastapi import FastAPI
from PIL import Image, ImageDraw, ImageFilter, ImageOps, Image as PILImage 
from PIL import Image
import numpy as np
import cv2
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi import UploadFile, File, Form
import io
from fastapi.responses import StreamingResponse
import os
import uuid
import datetime
import threading
import time
import logging
import logging.handlers
import traceback
import base64
import re
import gc
import asyncio
import psutil
import uvicorn

# -------- AI Models -------- #
try:
    from rembg import remove as rembg_remove
    REMBG_AVAILABLE = True
    print("✅ Rembg AI model loaded successfully")
except ImportError:
    REMBG_AVAILABLE = False
    print("❌ Rembg not available, using fallback methods")

# ----------------- Logging Setup ----------------- #
log_folder = 'logs'
os.makedirs(log_folder, exist_ok=True)
logger = logging.getLogger("AI_API")
logger.setLevel(logging.INFO)
handler = logging.handlers.RotatingFileHandler(
    filename=os.path.join(log_folder, "app.log"),
    maxBytes=5*1024*1024,
    backupCount=5
)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# ----------------- Folders ----------------- #
UPLOAD_FOLDER = 'uploads'
REMOVEBG_FOLDER = 'removebg'
VECTOR_FOLDER = 'vectorized'
STATIC_FOLDER = 'static'
for folder in [UPLOAD_FOLDER, REMOVEBG_FOLDER, VECTOR_FOLDER, STATIC_FOLDER]:
    os.makedirs(folder, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'webp', 'gif'}

# ----------------- Memory Management System ----------------- #
class MemoryManager:
    def __init__(self, cleanup_interval=300):
        self.cleanup_interval = cleanup_interval
        self.is_running = False
    
    async def start(self):
        """Start automatic memory cleanup"""
        if not self.is_running:
            self.is_running = True
            asyncio.create_task(self.periodic_cleanup())
            logger.info(f"🚀 Memory Manager Started - Cleaning every {self.cleanup_interval} seconds")
    
    async def periodic_cleanup(self):
        """Periodic memory cleanup task"""
        while self.is_running:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self.clean_memory()
            except Exception as e:
                logger.error(f"Memory cleanup error: {e}")
    
    async def clean_memory(self):
        """Perform memory cleanup"""
        try:
            process = psutil.Process(os.getpid())
            before_memory = process.memory_info().rss / 1024 / 1024
            
            collected = gc.collect()
            await self.clear_image_caches()
            
            after_memory = process.memory_info().rss / 1024 / 1024
            memory_freed = before_memory - after_memory
            
            logger.info(f"🧹 Memory Cleaned: {before_memory:.1f}MB → {after_memory:.1f}MB | Freed: {memory_freed:.1f}MB | Objects: {collected}")
            
        except Exception as e:
            logger.error(f"Memory cleaning failed: {e}")
    
    async def clear_image_caches(self):
        """Clear any image processing related caches"""
        try:
            if 'image_cache' in globals():
                globals()['image_cache'].clear()
        except Exception as e:
            logger.warning(f"Cache clearing warning: {e}")

# Initialize Memory Manager
memory_manager = MemoryManager(cleanup_interval=300)

# ----------------- FastAPI App ----------------- #
app = FastAPI(title="AI Image Processing API (BG Removal & SVG Converter)")

REMOVEBG_FOLDER = "/tmp"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup Event
@app.on_event("startup")
async def startup_event():
    """Start memory management on app startup"""
    await memory_manager.start()
    logger.info("✅ AI Image Processing API Started with Memory Management")

# Memory Management Endpoints
@app.get("/memory/clean")
async def manual_memory_clean():
    """Manual memory cleanup endpoint"""
    await memory_manager.clean_memory()
    return {"message": "Memory cleaned manually", "timestamp": datetime.datetime.now().isoformat()}

@app.get("/memory/status")
async def memory_status():
    """Get current memory status"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    return {
        "memory_usage_mb": round(memory_info.rss / 1024 / 1024, 2),
        "virtual_memory_mb": round(memory_info.vms / 1024 / 1024, 2),
        "timestamp": datetime.datetime.now().isoformat(),
        "auto_cleanup_active": True,
        "cleanup_interval_minutes": 5
    }

# ----------------------------------------------------
# --- Core SVG Logic Class ---
# ----------------------------------------------------
class SVGProcessor:
    def __init__(self):
        self.resampling_method = PILImage.Resampling.LANCZOS if hasattr(PILImage, 'Resampling') else PILImage.LANCZOS

    async def colorful_svg_conversion(self, image_data, simplification: int, color_palette_size: int) -> str:
        """Convert image to a colorful vector-like SVG"""
        try:
            image = PILImage.open(io.BytesIO(image_data))
            image = image.convert('RGB')
            
            if color_palette_size > 0 and color_palette_size < 256:
                quantized_image = image.quantize(colors=color_palette_size).convert('RGB')
            else:
                quantized_image = image
            
            processed_image = quantized_image
            if simplification > 0:
                processed_image = self._safe_simplify(quantized_image, simplification)
            
            max_size = 1200
            if max(processed_image.size) > max_size:
                processed_image.thumbnail((max_size, max_size), self.resampling_method)
            
            buffered = io.BytesIO()
            processed_image.save(buffered, format="PNG", optimize=True)
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            final_width, final_height = processed_image.size
            svg_content = f'''<svg width="{final_width}" height="{final_height}" viewBox="0 0 {final_width} {final_height}" xmlns="http://www.w3.org/2000/svg">
                <image href="data:image/png;base64,{img_str}" width="100%" height="100%"/>
            </svg>'''
            
            return svg_content
            
        except (IOError, OSError, ValueError) as e:
            logger.error(f"PIL/Image Processing ERROR in colorful_svg: {e}") 
            raise HTTPException(status_code=422, detail=f"Image processing failed. Is the file valid? Error: {e}")
        except Exception as e:
            logger.error(f"UNEXPECTED ERROR in colorful_svg: {e}")
            raise HTTPException(status_code=500, detail="Internal Server Error during conversion.")

    def _safe_simplify(self, image, simplification_level):
        """Safe image simplification without causing filter size errors"""
        try:
            if simplification_level <= 3:
                return image.filter(ImageFilter.SMOOTH)
            elif simplification_level <= 6:
                return image.filter(ImageFilter.SMOOTH_MORE)
            else:
                blur_radius = min(2.0, (simplification_level - 6) * 0.5)
                return image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        except Exception as e:
            logger.error(f"Image simplification failed: {e}")
            return image

    async def threshold_svg_conversion(self, image_data, threshold: int, stroke_color: str) -> str:
        """Threshold-based conversion for black and white vector effect"""
        try:
            image = PILImage.open(io.BytesIO(image_data))
            image = image.convert('RGB')
            
            grayscale_image = image.convert('L')
            threshold_image = grayscale_image.point(lambda x: 0 if x < threshold else 255)
            
            width, height = image.size
            final_image = PILImage.new('RGB', (width, height), color='white')
            
            stroke_rgb = tuple(int(stroke_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
            
            pixels = threshold_image.load()
            final_pixels = final_image.load()
            
            for x in range(width):
                for y in range(height):
                    if pixels[x, y] == 0:
                        final_pixels[x, y] = stroke_rgb
            
            max_size = 750
            if max(final_image.size) > max_size:
                final_image.thumbnail((max_size, max_size), self.resampling_method)
            
            buffered = io.BytesIO()
            final_image.save(buffered, format="PNG", optimize=True)
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            final_width, final_height = final_image.size
            svg_content = f'''<svg width="{final_width}" height="{final_height}" viewBox="0 0 {final_width} {final_height}" xmlns="http://www.w3.org/2000/svg">
                <image href="data:image/png;base64,{img_str}" width="100%" height="100%"/>
            </svg>'''
            
            return svg_content
            
        except Exception as e:
            logger.error(f"Threshold conversion error: {e}")
            raise HTTPException(status_code=500, detail=f"Threshold processing failed: {str(e)}")

# Instantiate the SVG Processor
svg_processor = SVGProcessor() 

# ----------------- Utility Functions (BG Removal) ----------------- #
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ----------------- Fallback Background Removal ----------------- #
def fallback_background_remove(image: PILImage.Image) -> PILImage.Image:
    """
    Simple fallback method when AI is not available
    Uses basic color-based background removal
    """
    try:
        img_array = np.array(image.convert('RGBA'))
        
        # Simple background removal based on white background assumption
        white_threshold = 200
        background_mask = (img_array[:, :, 0] > white_threshold) & \
                         (img_array[:, :, 1] > white_threshold) & \
                         (img_array[:, :, 2] > white_threshold)
        
        # Create alpha channel
        alpha = np.where(background_mask, 0, 255).astype(np.uint8)
        img_array[:, :, 3] = alpha
        
        return PILImage.fromarray(img_array, 'RGBA')
        
    except Exception as e:
        logger.error(f"Fallback background removal failed: {e}")
        return image.convert("RGBA")

# 1. 🖼️ সাবজেক্টের সীমানা অনুযায়ী ক্রপ করা
def crop_to_subject(image: PILImage.Image) -> PILImage.Image:
    """Crops the image to the smallest bounding box containing non-transparent pixels."""
    try:
        if image.mode != 'RGBA':
            return image
            
        alpha = np.array(image)[:, :, 3]
        coords = np.argwhere(alpha > 0)
        
        if coords.size == 0:
            return image
            
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        padding = 10
        width, height = image.size
        
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(width, x_max + padding + 1)
        y_max = min(height, y_max + padding + 1)
        
        cropped_image = image.crop((x_min, y_min, x_max, y_max))
        
        logger.info(f"Image cropped from {width}x{height} to {cropped_image.size[0]}x{cropped_image.size[1]}.")
        return cropped_image
        
    except Exception as e:
        logger.error(f"Image cropping failed: {e}")
        return image

# 2. 🎨 উন্নত Refine Edge: রঙের দূষণ অপসারণ (Decontamination)
def decontaminate_foreground(image: PILImage.Image) -> PILImage.Image:
    """Advanced decontamination to remove background color bleed/fringing"""
    try:
        img_array = np.array(image, dtype=np.float32)
        color = img_array[:, :, :3]
        alpha = img_array[:, :, 3]

        edge_mask = (alpha > 0) & (alpha < 255)
        
        opaque_color = color.copy()
        opaque_color[alpha == 0] = 255 

        kernel_size = 11 
        if kernel_size % 2 == 0:
            kernel_size += 1

        blurred_color = cv2.GaussianBlur(opaque_color.astype(np.uint8), (kernel_size, kernel_size), 0).astype(np.float32)
        color[edge_mask] = blurred_color[edge_mask]
        
        img_array[:, :, :3] = color
        
        return PILImage.fromarray(img_array.astype(np.uint8), 'RGBA')
        
    except Exception as e:
        logger.error(f"Decontamination (Refine Edge) failed: {e}")
        return image

# 3. ✨ মাস্ক স্মুথিং
def refine_mask_smoothing(image: PILImage.Image) -> PILImage.Image:
    """Optimized alpha channel smoothing for cleaner, softer edges."""
    try:
        img_array = np.array(image)
        alpha = img_array[:, :, 3].astype(np.float32)
        
        kernel = np.ones((5, 5), np.uint8)
        alpha_uint8 = alpha.astype(np.uint8)
        
        alpha_cleaned = cv2.morphologyEx(alpha_uint8, cv2.MORPH_CLOSE, kernel, iterations=1)
        alpha_cleaned = cv2.morphologyEx(alpha_cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
        
        alpha_smoothed = cv2.GaussianBlur(alpha_cleaned.astype(np.float32), (5, 5), 0.5)
        
        img_array[:, :, 3] = np.clip(alpha_smoothed, 0, 255).astype(np.uint8)
        
        return PILImage.fromarray(img_array, 'RGBA')
        
    except Exception as e:
        logger.error(f"Mask smoothing failed: {e}")
        return image

# 4. ⚫ কালো ছায়া দূর করা
def clean_dark_artifacts(image: PILImage.Image) -> PILImage.Image:
    """Removes dark color bleed (shadow artifacts) from semi-transparent edges."""
    try:
        img_array = np.array(image)
        color = img_array[:, :, :3]
        alpha = img_array[:, :, 3]

        edge_mask = (alpha > 0) & (alpha < 255)
        
        dark_threshold = 40
        is_dark = (color[:, :, 0] < dark_threshold) & \
                  (color[:, :, 1] < dark_threshold) & \
                  (color[:, :, 2] < dark_threshold)
        
        pixels_to_remove = edge_mask & is_dark

        if np.any(pixels_to_remove):
            img_array[pixels_to_remove, 3] = 0
            logger.info(f"Cleaned {np.sum(pixels_to_remove)} dark edge pixels.")

        return PILImage.fromarray(img_array, 'RGBA')
        
    except Exception as e:
        logger.error(f"Dark artifact cleaning failed: {e}")
        return image

# --------- Main Background Removal Function (Optimized) --------- #
def remove_background_optimized(image: PILImage.Image, quality: str) -> PILImage.Image:
    """
    Main function that uses the best model and applies custom cleaning.
    quality='high' includes post-processing (Refine Edge).
    quality='standard' skips post-processing for speed.
    """
    try:
        # ✅ Ensure rembg AI model cache works inside Railway container
        import os
        os.environ["U2NET_HOME"] = "/tmp/u2net"
        os.makedirs("/tmp/u2net", exist_ok=True)

        # ✅ Use AI if available, otherwise fallback
        if REMBG_AVAILABLE:
            buf = io.BytesIO()
            image.convert("RGB").save(buf, format="PNG", quality=95)
            img_bytes = buf.getvalue()
            
            # ⚙️ Run rembg — remove 'session_name' for better auto model detection
            result_bytes = rembg_remove(
                img_bytes, 
                post_process_mask=True
            )

            # ✅ Safety check: rembg output must not be empty
            if not result_bytes or len(result_bytes) < 1000:
                raise RuntimeError("rembg returned empty or invalid bytes")

            # ✅ Open image safely and ensure RGBA
            result_img = PILImage.open(io.BytesIO(result_bytes))
            if result_img.mode != "RGBA":
                result_img = result_img.convert("RGBA")

        else:
            # ⚠️ Use fallback method if AI not available
            logger.info("Using fallback background removal (AI not available)")
            result_img = fallback_background_remove(image)
        
        # ✅ Conditional Post-Processing Pipeline (Refine Edge)
        if quality.lower() == 'high':
            logger.info("Applying advanced post-processing (Refine Edge).")
            result_img = refine_mask_smoothing(result_img)
            result_img = decontaminate_foreground(result_img)
            result_img = clean_dark_artifacts(result_img)
        else:
            logger.info("Skipping advanced post-processing (Standard quality).")
        
        # ✅ Final Crop
        result_img = crop_to_subject(result_img)
        
        return result_img

    except Exception as e:
        logger.error(f"Background removal failed: {e}")
        logger.error(traceback.format_exc())
        return image.convert("RGBA")


# ----------------- Routes ----------------- #
current_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(current_dir, "static")
# 1. 🖼️ Static Files কনফিগারেশন
app.mount("/static", StaticFiles(directory="static"), name="static")

# Root route - index.html serve করুন

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/")
async def read_root():
    return FileResponse("static/index.html")

@app.get("/index.html")
async def index_page():
    return FileResponse("static/index.html")

# All HTML page routes
@app.get("/compress_tool")
@app.get("/compress_tool.html")
async def compress_tool():
    return FileResponse("static/compress_tool.html")

@app.get("/image-to-vector.html")
async def image_to_vector():
    return FileResponse("static/image-to-vector.html")

@app.get("/jpg-to-png.html")
async def jpg_to_png():
    return FileResponse("static/jpg-to-png.html")

@app.get("/bg_remove.html")
async def bg_remove():
    return FileResponse("static/bg_remove.html")

@app.get("/image-to-pdf.html")
async def image_to_pdf():
    return FileResponse("static/image-to-pdf.html")

@app.get("/all-convert.html")
async def all_convert():
    return FileResponse("static/all-convert.html")

# @app.get("/bg-remove.html")
# async def bg_remove():
#     return FileResponse("static/bg-remove.html")

@app.get("/collage-maker.html")
async def collage_maker():
    return FileResponse("static/collage-maker.html")

@app.get("/crop.html")
async def crop_tool():
    return FileResponse("static/crop.html")

@app.get("/flip-image")
@app.get("/flip-image.html")
async def flip_tool():
    return FileResponse("static/flip-image.html")

@app.get("/gif-to-jpg.html")
async def gif_to_jpg():
    return FileResponse("static/gif-to-jpg.html")

@app.get("/heice-to-jpg.html")
async def heice_to_jpg():
    return FileResponse("static/heice-to-jpg.html")

@app.get("/image-to-text.html")
async def image_to_text():
    return FileResponse("static/image-to-text.html")

@app.get("/join-image")
@app.get("/join-image.html")
async def join_image_page():
    return FileResponse("static/join-image.html")

@app.get("/meme-generator.html")
async def meme_generator():
    return FileResponse("static/meme-generator.html")

@app.get("/mosaic.html")
async def mosaic_tool():
    return FileResponse("static/mosaic.html")

@app.get("/photoEdit.html")
@app.get("/photo-edit.html") 
@app.get("/photo-edit")
async def photo_edit():
    return FileResponse("static/photo-edit.html")

@app.get("/png-to-jpg.html")
async def png_to_jpg():
    return FileResponse("static/png-to-jpg.html")

@app.get("/watermark.html")
async def watermark_page():
    return FileResponse("static/watermark.html")

@app.get("/web-to-jpg.html")
async def web_to_jpg():
    return FileResponse("static/web-to-jpg.html")

@app.get("/resize.html")
async def resize_tool():
    return FileResponse("static/resize.html")

@app.get("/rotate-image.html")
async def rotate_image():
    return FileResponse("static/rotate-image.html")

# API Status
@app.get("/api-status")
async def api_status():
    return {
        "message": "🚀 AI Image Processing API running! (BG Removal & SVG)", 
        "ai_available": REMBG_AVAILABLE,
        "bg_removal_quality": "Optimized High Quality (Advanced Refine Edge) or Standard (AI only)",
        "svg_conversion": "Available",
        "memory_management": "Active (Auto-clean every 5 minutes)"
    }

# ----------------------------------------------------
# --- SVG Vectorization Routes ---
# ----------------------------------------------------

@app.post("/vectorize") 
async def convert_to_svg(
    file: UploadFile = File(...),
    simplification: int = Form(2, description="Simplification level (0-10, higher = more simplified)"),
    color_palette_size: int = Form(32, description="Number of colors in palette (0 = original colors)"),
    mode: str = Form("colorful", description="Conversion mode: 'colorful' or 'threshold'")
):
    """Convert uploaded image to SVG with different modes"""
    
    allowed_types = ["image/jpeg", "image/png", "image/webp", "image/gif"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}. Must be JPG, PNG, WebP, or GIF.")

    if not 0 <= simplification <= 10:
        raise HTTPException(status_code=400, detail="Simplification must be between 0 and 10.")

    if not 0 <= color_palette_size <= 256:
        raise HTTPException(status_code=400, detail="Color palette size must be between 0 and 256.")

    image_data = await file.read()
    
    try:
        if mode == "threshold":
            svg_result = await svg_processor.threshold_svg_conversion(
                image_data, 
                threshold=128,
                stroke_color="#000000"
            )
        else:
            svg_result = await svg_processor.colorful_svg_conversion(
                image_data, 
                simplification=simplification,
                color_palette_size=color_palette_size
            )
        
        return Response(content=svg_result, media_type="image/svg+xml")
    
    except Exception as e:
        logger.error(f"Vectorize general error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Conversion failed: {str(e)}")
    finally:
        await file.close()

@app.post("/vectorize/threshold")
async def threshold_vectorize(
    file: UploadFile = File(...),
    threshold: int = Form(128, description="Threshold value (0-255)"),
    stroke_color: str = Form("#000000", description="Stroke color in hex format")
):
    """Original threshold-based vectorization"""
    
    if not 0 <= threshold <= 255:
        raise HTTPException(status_code=400, detail="Threshold must be between 0 and 255.")

    if not re.fullmatch(r'^#([A-Fa-f0-9]{6})$', stroke_color):
        raise HTTPException(status_code=400, detail="Invalid stroke_color format. Must be a 6-digit hex code (e.g., #000000).")

    allowed_types = ["image/jpeg", "image/png", "image/webp"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")

    image_data = await file.read()
    
    try:
        svg_result = await svg_processor.threshold_svg_conversion(
            image_data, 
            threshold=threshold,
            stroke_color=stroke_color
        )
        return Response(content=svg_result, media_type="image/svg+xml")
    except Exception as e:
        logger.error(f"Vectorize threshold error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Threshold conversion failed: {str(e)}")
    finally:
        await file.close()

# ----------------------------------------------------
# --- BG Removal Routes ---
# ----------------------------------------------------
@app.get("/get-image/{filename}")
async def get_image(filename: str):
    """Serve the processed image for preview."""
    file_path = os.path.join(REMOVEBG_FOLDER, filename)
    if os.path.exists(file_path):
        if filename.lower().endswith('.png'):
            media_type = "image/png"
        else:
            media_type = "image/jpeg"
            
        return FileResponse(file_path, media_type=media_type)
        
    return JSONResponse({"error": "Preview file not found"}, status_code=404)

@app.post("/remove-bg")
async def remove_bg(
    image: UploadFile = File(...), 
    background_color: str = Form(default="transparent"),
    quality: str = Form(default="high", description="Processing quality: 'high' (Refine Edge + AI) or 'standard' (AI only).")
):
    """Remove background with working download + transparent PNG"""
    try:
        # ✅ Validate file type
        if not allowed_file(image.filename):
            return JSONResponse({"error": "File type not allowed"}, status_code=400)

        # ✅ Read image bytes
        file_bytes = await image.read()
        try:
            img = PILImage.open(io.BytesIO(file_bytes))
            img.verify()
            img = PILImage.open(io.BytesIO(file_bytes)).convert("RGBA")
        except Exception as e:
            return JSONResponse({"error": f"Invalid image file: {str(e)}"}, status_code=400)

        # ✅ AI Background Remove (your function)
        processed_img = remove_background_optimized(img, quality)

        # ✅ Handle background color
        bg_rgb = None
        output_format = "PNG"

        if background_color.startswith("#") and len(background_color) == 7:
            try:
                bg_rgb = tuple(int(background_color[i:i+2], 16) for i in (1, 3, 5))
            except:
                bg_rgb = None

        if bg_rgb:
            bg_img = PILImage.new("RGB", processed_img.size, bg_rgb)
            bg_img.paste(processed_img, mask=processed_img.split()[3])
            processed_img = bg_img.convert("RGB")
            output_format = "JPEG"

        # ✅ Save output in-memory (no temp file issue)
        output_bytes = io.BytesIO()
        if output_format == "PNG":
            processed_img.save(output_bytes, format="PNG", optimize=True)
        else:
            processed_img.save(output_bytes, format="JPEG", quality=95, optimize=True)
        output_bytes.seek(0)

        filename = f"nobg_{uuid.uuid4().hex}.{output_format.lower()}"
        file_path = os.path.join(REMOVEBG_FOLDER, filename)

        # Optional: keep a copy in /tmp for preview endpoint
        with open(file_path, "wb") as f:
            f.write(output_bytes.getbuffer())

        quality_message = (
            "Optimized quality with Refine Edge feature." 
            if quality.lower() == "high" 
            else "Standard quality (AI only)."
        )

        return {
            "success": True,
            "filename": filename,
            "previewUrl": f"/get-image/{filename}",
            "downloadUrl": f"/download/{filename}",
            "message": f"Background removed with {quality_message}",
            "ai_used": REMBG_AVAILABLE,
            "format": output_format,
            "background": background_color
        }

    except Exception as e:
        logger.error(f"Remove BG error: {traceback.format_exc()}")
        return JSONResponse({"error": f"Processing failed: {str(e)}"}, status_code=500)


# ✅ Download endpoint (works perfectly)
@app.get("/download/{filename}")
async def download_image(filename: str):
    file_path = os.path.join(REMOVEBG_FOLDER, filename)
    if not os.path.exists(file_path):
        return JSONResponse({"error": "File not found"}, status_code=404)
    return StreamingResponse(
        open(file_path, "rb"),
        media_type="image/png" if filename.lower().endswith(".png") else "image/jpeg",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

# ✅ Alias route (function-এর বাইরে)
@app.post("/api/remove-background")
async def alias_remove_background(
    image: UploadFile = File(...),
    background_color: str = Form("transparent"),
    quality: str = Form("high")
):
    """
    Alias route to maintain compatibility with older frontend URLs.
    Redirects /api/remove-background to /remove-bg internally.
    """
    return await remove_bg(image=image, background_color=background_color, quality=quality)

# ----------------- Background Cleanup ----------------- #
def cleanup_files():
    while True:
        try:
            now = datetime.datetime.now()
            for folder in [UPLOAD_FOLDER, REMOVEBG_FOLDER, VECTOR_FOLDER, STATIC_FOLDER]: 
                for fname in os.listdir(folder):
                    fpath = os.path.join(folder, fname)
                    if folder == STATIC_FOLDER and fname.lower() == 'index.html':
                        continue
                        
                    if os.path.isfile(fpath):
                        ctime = datetime.datetime.fromtimestamp(os.path.getctime(fpath))
                        if (now - ctime).total_seconds() > 3600:
                            os.remove(fpath)
                            logger.info(f"Removed old file: {fpath}")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
        time.sleep(1800)

threading.Thread(target=cleanup_files, daemon=True).start()

# ----------------- Run ----------------- #
if __name__ == "__main__":
    print("🚀 Starting AI Image Processing API...")
    print("=" * 60)
    print("✨ Available Services:")
    print("  1. Web Interface (Root: /)")
    print("  2. Background Removal (API: /remove-bg)")
    print("  3. SVG Vectorization (API: /vectorize)")
    print("  4. Memory Management (Auto-clean every 5 minutes)")
    print("=" * 60)
    print(f"🤖 AI Model Available: {'✅ Yes' if REMBG_AVAILABLE else '❌ No'}")
    print("🧠 Memory Management: ✅ Active")
    print(f"🌐 Server URL: https://free-production-0a3b.up.railway.app")
    print("=" * 60)
    
# ----------------- Run ----------------- #
if __name__ == "__main__":
    import os
    # Railway-এর দেওয়া PORT ভ্যারিয়েবলটি ব্যবহার করুন। যদি না পাওয়া যায়, তবে 8000 ব্যবহার করুন।
    port = int(os.environ.get("PORT", 8000))
    print(f"🌐 Server running on port: {port}")
    
    # uvicorn.run ফাংশনে এই পরিবর্তিত 'port' ভ্যারিয়েবলটি ব্যবহার করুন
    uvicorn.run("app:app", host="0.0.0.0", port=port)
