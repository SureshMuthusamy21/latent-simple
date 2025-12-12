"""
FastAPI server for LatentSync Lip-Sync API
Wraps the existing inference pipeline without modifying core logic
"""

import os
import uuid
import shutil
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
from omegaconf import OmegaConf
from diffusers import AutoencoderKL, DDIMScheduler
from accelerate.utils import set_seed

# Import from existing codebase
from latentsync.models.unet import UNet3DConditionModel
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
from latentsync.whisper.audio2feature import Audio2Feature
from DeepCache import DeepCacheSDHelper

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

class APIConfig:
    """API Configuration - matches inference.py defaults"""
    UNET_CONFIG_PATH = "configs/unet/stage2_512.yaml"
    INFERENCE_CKPT_PATH = "checkpoints/latentsync_unet.pt"
    TEMP_BASE_DIR = "temp"  # MUST match Gradio for correct pipeline processing
    MAX_VIDEO_SIZE_MB = 500
    MAX_AUDIO_SIZE_MB = 50
    ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".webm"}
    ALLOWED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac"}

# ============================================================================
# Inference Engine Singleton (Models loaded once at startup)
# ============================================================================

class InferenceEngine:
    """
    Singleton class that loads models once and reuses them for all requests.
    This is the key optimization - models take ~10-30s to load.
    """
    
    def __init__(self):
        self.pipeline = None
        self.config = None
        self.dtype = None
        self.initialized = False
        
    def initialize(
        self,
        unet_config_path: str = APIConfig.UNET_CONFIG_PATH,
        checkpoint_path: str = APIConfig.INFERENCE_CKPT_PATH
    ):
        """
        Initialize models once at startup.
        This code is extracted from inference.py lines 41-82
        """
        if self.initialized:
            logger.info("Engine already initialized, skipping...")
            return
            
        logger.info("🚀 Initializing Inference Engine...")
        
        # Check if checkpoint exists
        if not os.path.exists(checkpoint_path):
            raise RuntimeError(f"Checkpoint not found: {checkpoint_path}")
        if not os.path.exists(unet_config_path):
            raise RuntimeError(f"Config not found: {unet_config_path}")
        
        # Load configuration
        self.config = OmegaConf.load(unet_config_path)
        
        # Check GPU and set dtype
        is_fp16_supported = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 7
        self.dtype = torch.float16 if is_fp16_supported else torch.float32
        logger.info(f"Using dtype: {self.dtype}")
        
        # Load scheduler
        logger.info("Loading DDIMScheduler...")
        scheduler = DDIMScheduler.from_pretrained("configs")
        
        # Determine whisper model based on cross_attention_dim
        if self.config.model.cross_attention_dim == 768:
            whisper_model_path = "checkpoints/whisper/small.pt"
        elif self.config.model.cross_attention_dim == 384:
            whisper_model_path = "checkpoints/whisper/tiny.pt"
        else:
            raise NotImplementedError("cross_attention_dim must be 768 or 384")
        
        # Load audio encoder
        logger.info(f"Loading Audio2Feature with {whisper_model_path}...")
        audio_encoder = Audio2Feature(
            model_path=whisper_model_path,
            device="cuda",
            num_frames=self.config.data.num_frames,
            audio_feat_length=self.config.data.audio_feat_length,
        )
        
        # Load VAE
        logger.info("Loading VAE...")
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=self.dtype)
        vae.config.scaling_factor = 0.18215
        vae.config.shift_factor = 0
        
        # Load UNet
        logger.info(f"Loading UNet from {checkpoint_path}...")
        unet, _ = UNet3DConditionModel.from_pretrained(
            OmegaConf.to_container(self.config.model),
            checkpoint_path,
            device="cpu",
        )
        unet = unet.to(dtype=self.dtype)
        
        # Create pipeline
        logger.info("Creating LipsyncPipeline...")
        self.pipeline = LipsyncPipeline(
            vae=vae,
            audio_encoder=audio_encoder,
            unet=unet,
            scheduler=scheduler,
        ).to("cuda")
        
        self.initialized = True
        logger.info("✅ Inference Engine initialized successfully!")
        
    def process(
        self,
        video_path: str,
        audio_path: str,
        video_out_path: str,
        inference_steps: int = 20,
        guidance_scale: float = 1.0,
        seed: int = 1247,
        enable_deepcache: bool = False,
        temp_dir: str = "temp"
    ):
        """
        Run inference using the loaded pipeline.
        This code is extracted from inference.py lines 84-103
        """
        if not self.initialized:
            raise RuntimeError("Engine not initialized. Call initialize() first.")
        
        # Validate inputs
        if not os.path.exists(video_path):
            raise RuntimeError(f"Video path '{video_path}' not found")
        if not os.path.exists(audio_path):
            raise RuntimeError(f"Audio path '{audio_path}' not found")
        
        logger.info(f"Processing: video={video_path}, audio={audio_path}")
        
        # Optional: Enable DeepCache
        helper = None
        if enable_deepcache:
            logger.info("Enabling DeepCache...")
            helper = DeepCacheSDHelper(pipe=self.pipeline)
            helper.set_params(cache_interval=3, cache_branch_id=0)
            helper.enable()
        
        # Set seed
        if seed != -1:
            set_seed(seed)
        else:
            torch.seed()
        
        logger.info(f"Seed: {torch.initial_seed()}")
        
        # Run pipeline - this is the actual inference
        self.pipeline(
            video_path=video_path,
            audio_path=audio_path,
            video_out_path=video_out_path,
            num_frames=self.config.data.num_frames,
            num_inference_steps=inference_steps,
            guidance_scale=guidance_scale,
            weight_dtype=self.dtype,
            width=self.config.data.resolution,
            height=self.config.data.resolution,
            mask_image_path=self.config.data.mask_image_path,
            temp_dir=temp_dir,
        )
        
        # Disable DeepCache if enabled
        if helper:
            helper.disable()
        
        logger.info(f"✅ Processing complete: {video_out_path}")

# Global inference engine instance
engine = InferenceEngine()

# ============================================================================
# Lifespan Context Manager (replaces deprecated on_event)
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize models once at server startup"""
    logger.info("🌟 Starting LatentSync API Server...")
    
    # Create temp directory
    os.makedirs(APIConfig.TEMP_BASE_DIR, exist_ok=True)
    
    # Initialize inference engine
    try:
        engine.initialize(
            unet_config_path=APIConfig.UNET_CONFIG_PATH,
            checkpoint_path=APIConfig.INFERENCE_CKPT_PATH
        )
    except Exception as e:
        logger.error(f"❌ Failed to initialize engine: {e}")
        raise
    
    yield  # Server runs here
    
    # Cleanup on shutdown (optional)
    logger.info("🛑 Shutting down LatentSync API Server...")

# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="LatentSync Lip-Sync API",
    description="AI-powered lip-sync generation API",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Helper Functions
# ============================================================================

def validate_file_extension(filename: str, allowed_extensions: set) -> bool:
    """Validate file extension"""
    return Path(filename).suffix.lower() in allowed_extensions

def validate_file_size(file: UploadFile, max_size_mb: int) -> bool:
    """Validate file size"""
    file.file.seek(0, 2)  # Seek to end
    size_mb = file.file.tell() / (1024 * 1024)
    file.file.seek(0)  # Reset
    return size_mb <= max_size_mb

async def save_upload_file(upload_file: UploadFile, destination: Path) -> None:
    """Save uploaded file to destination"""
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        with destination.open("wb") as buffer:
            content = await upload_file.read()  # Read async
            buffer.write(content)
        
        # Verify file was saved
        if not destination.exists():
            raise RuntimeError(f"Failed to save file to {destination}")
        
        file_size = destination.stat().st_size
        logger.info(f"Saved {destination.name}: {file_size / 1024:.2f} KB")
        
    except Exception as e:
        logger.error(f"Error saving file to {destination}: {e}")
        raise
    finally:
        await upload_file.seek(0)  # Reset for potential reuse

def cleanup_temp_directory(directory: str):
    """Clean up temporary directory"""
    try:
        if os.path.exists(directory):
            shutil.rmtree(directory)
            logger.info(f"🧹 Cleaned up temp directory: {directory}")
    except Exception as e:
        logger.warning(f"Failed to cleanup {directory}: {e}")

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "engine_initialized": engine.initialized,
        "gpu_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
    }

@app.post("/api/v1/lipsync")
async def create_lipsync(
    video: UploadFile = File(..., description="Input video file"),
    audio: UploadFile = File(..., description="Input audio file"),
    inference_steps: int = Form(20, description="Number of inference steps (default: 20)"),
    guidance_scale: float = Form(1.0, description="Guidance scale (default: 1.0)"),
    seed: int = Form(1247, description="Random seed (-1 for random)"),
    enable_deepcache: bool = Form(False, description="Enable DeepCache optimization")
):
    """
    Generate lip-synced video from input video and audio.
    
    Returns the processed video file directly for download.
    """
    request_id = str(uuid.uuid4())[:8]
    request_dir = None
    
    try:
        # ========================================================================
        # 1. Validate inputs
        # ========================================================================
        
        # Validate video
        if not validate_file_extension(video.filename, APIConfig.ALLOWED_VIDEO_EXTENSIONS):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid video format. Allowed: {APIConfig.ALLOWED_VIDEO_EXTENSIONS}"
            )
        
        if not validate_file_size(video, APIConfig.MAX_VIDEO_SIZE_MB):
            raise HTTPException(
                status_code=400,
                detail=f"Video file too large. Max size: {APIConfig.MAX_VIDEO_SIZE_MB}MB"
            )
        
        # Validate audio
        if not validate_file_extension(audio.filename, APIConfig.ALLOWED_AUDIO_EXTENSIONS):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid audio format. Allowed: {APIConfig.ALLOWED_AUDIO_EXTENSIONS}"
            )
        
        if not validate_file_size(audio, APIConfig.MAX_AUDIO_SIZE_MB):
            raise HTTPException(
                status_code=400,
                detail=f"Audio file too large. Max size: {APIConfig.MAX_AUDIO_SIZE_MB}MB"
            )
        
        # ========================================================================
        # 2. Create temporary directory for this request (ABSOLUTE PATH)
        # ========================================================================
        
        # Use absolute path to avoid path resolution issues
        base_dir = Path(APIConfig.TEMP_BASE_DIR).resolve()
        request_dir = base_dir / request_id
        request_dir.mkdir(parents=True, exist_ok=True)
        
        # Define paths with absolute paths
        video_path = request_dir / f"input_video{Path(video.filename).suffix}"
        audio_path = request_dir / f"input_audio{Path(audio.filename).suffix}"
        output_path = request_dir / "output.mp4"
        
        logger.info(f"[{request_id}] Request directory: {request_dir}")
        logger.info(f"[{request_id}] Video path: {video_path}")
        logger.info(f"[{request_id}] Audio path: {audio_path}")
        
        # ========================================================================
        # 3. Save uploaded files
        # ========================================================================
        
        logger.info(f"[{request_id}] Saving uploaded files...")
        await save_upload_file(video, video_path)
        await save_upload_file(audio, audio_path)
        
        # Double-check files exist
        if not video_path.exists():
            raise HTTPException(
                status_code=500,
                detail=f"Failed to save video file to {video_path}"
            )
        if not audio_path.exists():
            raise HTTPException(
                status_code=500,
                detail=f"Failed to save audio file to {audio_path}"
            )
        
        # ========================================================================
        # 4. Run inference in thread pool (blocking operation)
        # ========================================================================
        
        logger.info(f"[{request_id}] Starting inference...")
        
        def run_inference():
            # Use absolute paths as strings (native Windows format)
            # CRITICAL: temp_dir must be "temp" to match Gradio and avoid artifacts
            engine.process(
                video_path=str(video_path.absolute()),
                audio_path=str(audio_path.absolute()),
                video_out_path=str(output_path.absolute()),
                inference_steps=inference_steps,
                guidance_scale=guidance_scale,
                seed=seed,
                enable_deepcache=enable_deepcache,
                temp_dir="temp"  # Simple relative path like Gradio
            )
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, run_inference)
        
        # ========================================================================
        # 5. Verify output exists
        # ========================================================================
        
        if not output_path.exists():
            raise HTTPException(
                status_code=500,
                detail="Inference completed but output file not found"
            )
        
        # ========================================================================
        # 6. Return file response
        # ========================================================================
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"lipsync_output_{timestamp}.mp4"
        
        logger.info(f"[{request_id}] ✅ Returning output file: {output_filename}")
        
        return FileResponse(
            path=str(output_path.absolute()),
            media_type="video/mp4",
            filename=output_filename,
            background=None  # Don't delete yet, will cleanup later
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
        
    except Exception as e:
        logger.error(f"[{request_id}] ❌ Error during processing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
        
    finally:
        # Cleanup temp directory after a delay (5 minutes)
        # This allows the FileResponse to complete
        if request_dir:
            async def delayed_cleanup():
                await asyncio.sleep(300)  # 5 minutes
                cleanup_temp_directory(str(request_dir))
            
            asyncio.create_task(delayed_cleanup())

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "LatentSync Lip-Sync API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

# ============================================================================
# Run Server (for development)
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Set to True for development
        log_level="info"
    )