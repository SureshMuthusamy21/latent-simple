# """
# FastAPI server for LatentSync Lip-Sync API
# Wraps the existing inference pipeline without modifying core logic
# """

# import os
# import uuid
# import shutil
# import asyncio
# from datetime import datetime
# from pathlib import Path
# from typing import Optional
# import logging
# from contextlib import asynccontextmanager

# from fastapi import FastAPI, File, UploadFile, HTTPException, Form
# from fastapi.responses import FileResponse, JSONResponse
# from fastapi.middleware.cors import CORSMiddleware
# import torch
# from omegaconf import OmegaConf
# from diffusers import AutoencoderKL, DDIMScheduler
# from accelerate.utils import set_seed
# import boto3

# # Import from existing codebase
# from latentsync.models.unet import UNet3DConditionModel
# from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
# from latentsync.whisper.audio2feature import Audio2Feature
# from DeepCache import DeepCacheSDHelper

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # ============================================================================
# # Configuration
# # ============================================================================

# class APIConfig:
#     """API Configuration - matches inference.py defaults"""
#     UNET_CONFIG_PATH = "configs/unet/stage2_512.yaml"
#     INFERENCE_CKPT_PATH = "checkpoints/latentsync_unet.pt"
#     TEMP_BASE_DIR = "temp_api"  # CRITICAL: Different from 'temp' used by pipeline!
#     MAX_VIDEO_SIZE_MB = 500
#     MAX_AUDIO_SIZE_MB = 50
#     ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".webm"}
#     ALLOWED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac"}
#     S3_BUCKET_NAME = "prod-video-gen-avatar-storage"
#     S3_VIDEO_KEY = "reya.mp4"

# # ============================================================================
# # S3 Helper
# # ============================================================================

# def download_video_from_s3(bucket_name: str, key: str, destination: Path) -> None:
#     """Download video from S3"""
#     try:
#         logger.info(f"Downloading {key} from S3 bucket {bucket_name}...")
#         s3_client = boto3.client('s3')
#         destination.parent.mkdir(parents=True, exist_ok=True)
#         s3_client.download_file(bucket_name, key, str(destination))
        
#         if not destination.exists():
#             raise RuntimeError(f"Failed to download file from S3")
        
#         file_size = destination.stat().st_size
#         logger.info(f"Downloaded {destination.name}: {file_size / 1024:.2f} KB")
        
#     except Exception as e:
#         logger.error(f"Error downloading from S3: {e}")
#         raise

# # ============================================================================
# # Inference Engine Singleton (Models loaded once at startup)
# # ============================================================================

# class InferenceEngine:
#     """
#     Singleton class that loads models once and reuses them for all requests.
#     This is the key optimization - models take ~10-30s to load.
#     """
    
#     def __init__(self):
#         self.pipeline = None
#         self.config = None
#         self.dtype = None
#         self.initialized = False
        
#     def initialize(
#         self,
#         unet_config_path: str = APIConfig.UNET_CONFIG_PATH,
#         checkpoint_path: str = APIConfig.INFERENCE_CKPT_PATH
#     ):
#         """
#         Initialize models once at startup.
#         This code is extracted from inference.py lines 41-82
#         """
#         if self.initialized:
#             logger.info("Engine already initialized, skipping...")
#             return
            
#         logger.info("🚀 Initializing Inference Engine...")
        
#         # Check if checkpoint exists
#         if not os.path.exists(checkpoint_path):
#             raise RuntimeError(f"Checkpoint not found: {checkpoint_path}")
#         if not os.path.exists(unet_config_path):
#             raise RuntimeError(f"Config not found: {unet_config_path}")
        
#         # Load configuration
#         self.config = OmegaConf.load(unet_config_path)
        
#         # Check GPU and set dtype
#         is_fp16_supported = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 7
#         self.dtype = torch.float16 if is_fp16_supported else torch.float32
#         logger.info(f"Using dtype: {self.dtype}")
        
#         # Load scheduler
#         logger.info("Loading DDIMScheduler...")
#         scheduler = DDIMScheduler.from_pretrained("configs")
        
#         # Determine whisper model based on cross_attention_dim
#         if self.config.model.cross_attention_dim == 768:
#             whisper_model_path = "checkpoints/whisper/small.pt"
#         elif self.config.model.cross_attention_dim == 384:
#             whisper_model_path = "checkpoints/whisper/tiny.pt"
#         else:
#             raise NotImplementedError("cross_attention_dim must be 768 or 384")
        
#         # Load audio encoder
#         logger.info(f"Loading Audio2Feature with {whisper_model_path}...")
#         audio_encoder = Audio2Feature(
#             model_path=whisper_model_path,
#             device="cuda",
#             num_frames=self.config.data.num_frames,
#             audio_feat_length=self.config.data.audio_feat_length,
#         )
        
#         # Load VAE
#         logger.info("Loading VAE...")
#         vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=self.dtype)
#         vae.config.scaling_factor = 0.18215
#         vae.config.shift_factor = 0
        
#         # Load UNet
#         logger.info(f"Loading UNet from {checkpoint_path}...")
#         unet, _ = UNet3DConditionModel.from_pretrained(
#             OmegaConf.to_container(self.config.model),
#             checkpoint_path,
#             device="cpu",
#         )
#         unet = unet.to(dtype=self.dtype)
        
#         # Create pipeline
#         logger.info("Creating LipsyncPipeline...")
#         self.pipeline = LipsyncPipeline(
#             vae=vae,
#             audio_encoder=audio_encoder,
#             unet=unet,
#             scheduler=scheduler,
#         ).to("cuda")
        
#         self.initialized = True
#         logger.info("✅ Inference Engine initialized successfully!")
        
#     def process(
#         self,
#         video_path: str,
#         audio_path: str,
#         video_out_path: str,
#         inference_steps: int = 20,
#         guidance_scale: float = 1.0,
#         seed: int = 1247,
#         enable_deepcache: bool = False,
#         temp_dir: str = "temp"
#     ):
#         """
#         Run inference using the loaded pipeline.
#         This code is extracted from inference.py lines 84-103
#         """
#         if not self.initialized:
#             raise RuntimeError("Engine not initialized. Call initialize() first.")
        
#         # Validate inputs
#         if not os.path.exists(video_path):
#             raise RuntimeError(f"Video path '{video_path}' not found")
#         if not os.path.exists(audio_path):
#             raise RuntimeError(f"Audio path '{audio_path}' not found")
        
#         logger.info(f"Processing: video={video_path}, audio={audio_path}")
        
#         # Optional: Enable DeepCache
#         helper = None
#         if enable_deepcache:
#             logger.info("Enabling DeepCache...")
#             helper = DeepCacheSDHelper(pipe=self.pipeline)
#             helper.set_params(cache_interval=3, cache_branch_id=0)
#             helper.enable()
        
#         # Set seed
#         if seed != -1:
#             set_seed(seed)
#         else:
#             torch.seed()
        
#         logger.info(f"Seed: {torch.initial_seed()}")
        
#         # Run pipeline - this is the actual inference
#         self.pipeline(
#             video_path=video_path,
#             audio_path=audio_path,
#             video_out_path=video_out_path,
#             num_frames=self.config.data.num_frames,
#             num_inference_steps=inference_steps,
#             guidance_scale=guidance_scale,
#             weight_dtype=self.dtype,
#             width=self.config.data.resolution,
#             height=self.config.data.resolution,
#             mask_image_path=self.config.data.mask_image_path,
#             temp_dir=temp_dir,
#         )
        
#         # Disable DeepCache if enabled
#         if helper:
#             helper.disable()
        
#         logger.info(f"✅ Processing complete: {video_out_path}")

# # Global inference engine instance
# engine = InferenceEngine()

# # ============================================================================
# # Lifespan Context Manager (replaces deprecated on_event)
# # ============================================================================

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     """Initialize models once at server startup"""
#     logger.info("🌟 Starting LatentSync API Server...")
    
#     # Create temp directory
#     os.makedirs(APIConfig.TEMP_BASE_DIR, exist_ok=True)
    
#     # Initialize inference engine
#     try:
#         engine.initialize(
#             unet_config_path=APIConfig.UNET_CONFIG_PATH,
#             checkpoint_path=APIConfig.INFERENCE_CKPT_PATH
#         )
#     except Exception as e:
#         logger.error(f"❌ Failed to initialize engine: {e}")
#         raise
    
#     yield  # Server runs here
    
#     # Cleanup on shutdown (optional)
#     logger.info("🛑 Shutting down LatentSync API Server...")

# # ============================================================================
# # FastAPI Application
# # ============================================================================

# app = FastAPI(
#     title="LatentSync Lip-Sync API",
#     description="AI-powered lip-sync generation API",
#     version="1.0.0",
#     lifespan=lifespan
# )

# # Add CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Configure appropriately for production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # ============================================================================
# # Helper Functions
# # ============================================================================

# def validate_file_extension(filename: str, allowed_extensions: set) -> bool:
#     """Validate file extension"""
#     return Path(filename).suffix.lower() in allowed_extensions

# def validate_file_size(file: UploadFile, max_size_mb: int) -> bool:
#     """Validate file size"""
#     file.file.seek(0, 2)  # Seek to end
#     size_mb = file.file.tell() / (1024 * 1024)
#     file.file.seek(0)  # Reset
#     return size_mb <= max_size_mb

# async def save_upload_file(upload_file: UploadFile, destination: Path) -> None:
#     """Save uploaded file to destination"""
#     try:
#         destination.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
#         with destination.open("wb") as buffer:
#             content = await upload_file.read()  # Read async
#             buffer.write(content)
        
#         # Verify file was saved
#         if not destination.exists():
#             raise RuntimeError(f"Failed to save file to {destination}")
        
#         file_size = destination.stat().st_size
#         logger.info(f"Saved {destination.name}: {file_size / 1024:.2f} KB")
        
#     except Exception as e:
#         logger.error(f"Error saving file to {destination}: {e}")
#         raise
#     finally:
#         await upload_file.seek(0)  # Reset for potential reuse

# def cleanup_temp_directory(directory: str):
#     """Clean up temporary directory"""
#     try:
#         if os.path.exists(directory):
#             shutil.rmtree(directory)
#             logger.info(f"🧹 Cleaned up temp directory: {directory}")
#     except Exception as e:
#         logger.warning(f"Failed to cleanup {directory}: {e}")

# # ============================================================================
# # API Endpoints
# # ============================================================================

# @app.get("/health")
# async def health_check():
#     """Health check endpoint"""
#     return {
#         "status": "healthy",
#         "engine_initialized": engine.initialized,
#         "gpu_available": torch.cuda.is_available(),
#         "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
#     }

# @app.post("/api/v1/lipsync")
# async def create_lipsync(
#     audio: UploadFile = File(..., description="Input audio file"),
#     inference_steps: int = Form(20, description="Number of inference steps (default: 20)"),
#     guidance_scale: float = Form(1.0, description="Guidance scale (default: 1.0)"),
#     seed: int = Form(1247, description="Random seed (-1 for random)"),
#     enable_deepcache: bool = Form(False, description="Enable DeepCache optimization")
# ):
#     """
#     Generate lip-synced video from S3 video and uploaded audio.
#     Returns the processed video file directly for download.
#     """
#     request_id = str(uuid.uuid4())[:8]
#     temp_files = []
    
#     try:
#         # ========================================================================
#         # 1. Validate audio input
#         # ========================================================================
        
#         if not validate_file_extension(audio.filename, APIConfig.ALLOWED_AUDIO_EXTENSIONS):
#             raise HTTPException(
#                 status_code=400,
#                 detail=f"Invalid audio format. Allowed: {APIConfig.ALLOWED_AUDIO_EXTENSIONS}"
#             )
        
#         if not validate_file_size(audio, APIConfig.MAX_AUDIO_SIZE_MB):
#             raise HTTPException(
#                 status_code=400,
#                 detail=f"Audio file too large. Max size: {APIConfig.MAX_AUDIO_SIZE_MB}MB"
#             )
        
#         # ========================================================================
#         # 2. Setup paths
#         # ========================================================================
        
#         base_dir = Path(APIConfig.TEMP_BASE_DIR).resolve()
#         base_dir.mkdir(parents=True, exist_ok=True)
        
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         video_path = base_dir / f"{request_id}_{timestamp}_input.mp4"
#         audio_path = base_dir / f"{request_id}_{timestamp}_audio{Path(audio.filename).suffix}"
#         output_path = base_dir / f"{request_id}_{timestamp}_output.mp4"
        
#         temp_files = [video_path, audio_path, output_path]
        
#         logger.info(f"[{request_id}] Video path: {video_path}")
#         logger.info(f"[{request_id}] Audio path: {audio_path}")
#         logger.info(f"[{request_id}] Output path: {output_path}")
        
#         # ========================================================================
#         # 3. Download video from S3 and save audio
#         # ========================================================================
        
#         logger.info(f"[{request_id}] Downloading video from S3...")
#         loop = asyncio.get_event_loop()
#         await loop.run_in_executor(
#             None,
#             download_video_from_s3,
#             APIConfig.S3_BUCKET_NAME,
#             APIConfig.S3_VIDEO_KEY,
#             video_path
#         )
        
#         logger.info(f"[{request_id}] Saving uploaded audio...")
#         await save_upload_file(audio, audio_path)
        
#         # Verify files exist
#         if not video_path.exists():
#             raise HTTPException(status_code=500, detail=f"Failed to download video from S3")
#         if not audio_path.exists():
#             raise HTTPException(status_code=500, detail=f"Failed to save audio file")
        
#         # ========================================================================
#         # 4. Run inference
#         # ========================================================================
        
#         logger.info(f"[{request_id}] Starting inference...")
        
#         def run_inference():
#             engine.process(
#                 video_path=str(video_path),
#                 audio_path=str(audio_path),
#                 video_out_path=str(output_path),
#                 inference_steps=inference_steps,
#                 guidance_scale=guidance_scale,
#                 seed=seed,
#                 enable_deepcache=enable_deepcache,
#                 temp_dir="temp"
#             )
        
#         await loop.run_in_executor(None, run_inference)
        
#         # ========================================================================
#         # 5. Verify output exists
#         # ========================================================================
        
#         if not output_path.exists():
#             raise HTTPException(
#                 status_code=500,
#                 detail="Inference completed but output file not found"
#             )
        
#         # ========================================================================
#         # 6. Return file response
#         # ========================================================================
        
#         output_filename = f"lipsync_output_{timestamp}.mp4"
        
#         logger.info(f"[{request_id}] ✅ Returning output file: {output_filename}")
        
#         return FileResponse(
#             path=str(output_path),
#             media_type="video/mp4",
#             filename=output_filename,
#             background=None
#         )
        
#     except HTTPException:
#         raise
        
#     except Exception as e:
#         logger.error(f"[{request_id}] ❌ Error: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
        
#     finally:
#         # Cleanup individual files after delay
#         if temp_files:
#             async def delayed_cleanup():
#                 await asyncio.sleep(300)  # 5 minutes
#                 for file_path in temp_files:
#                     try:
#                         if file_path.exists():
#                             file_path.unlink()
#                             logger.info(f"🧹 Deleted: {file_path.name}")
#                     except Exception as e:
#                         logger.warning(f"Cleanup failed for {file_path.name}: {e}")
            
#             asyncio.create_task(delayed_cleanup())

# @app.get("/")
# async def root():
#     """Root endpoint"""
#     return {
#         "message": "LatentSync Lip-Sync API",
#         "version": "1.0.0",
#         "docs": "/docs",
#         "health": "/health"
#     }

# # ============================================================================
# # Run Server (for development)
# # ============================================================================

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(
#         "workingmain:app",
#         host="0.0.0.0",
#         port=8000,
#         reload=False,  # Set to True for development
#         log_level="info"
#     )








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
import io

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
from omegaconf import OmegaConf
from diffusers import AutoencoderKL, DDIMScheduler
from accelerate.utils import set_seed
import boto3
import docx

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
    TEMP_BASE_DIR = "temp_api"  # CRITICAL: Different from 'temp' used by pipeline!
    MAX_VIDEO_SIZE_MB = 500
    MAX_AUDIO_SIZE_MB = 50
    ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".webm"}
    ALLOWED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac"}
    S3_BUCKET_NAME = "prod-video-gen-avatar-storage"
    AWS_REGION = "us-east-1"  # AWS region for S3 and Polly

# ============================================================================
# S3 Helper
# ============================================================================

def download_video_from_s3(bucket_name: str, key: str, destination: Path) -> None:
    """Download video from S3"""
    try:
        logger.info(f"Downloading {key} from S3 bucket {bucket_name}...")
        s3_client = boto3.client('s3', region_name=APIConfig.AWS_REGION)
        destination.parent.mkdir(parents=True, exist_ok=True)
        s3_client.download_file(bucket_name, key, str(destination))
        
        if not destination.exists():
            raise RuntimeError(f"Failed to download file from S3")
        
        file_size = destination.stat().st_size
        logger.info(f"Downloaded {destination.name}: {file_size / 1024:.2f} KB")
        
    except Exception as e:
        logger.error(f"Error downloading from S3: {e}")
        raise

# ============================================================================
# Document and Polly Helpers
# ============================================================================

def extract_text_from_docx(file_content: bytes) -> str:
    """Extract text from Word document"""
    try:
        logger.info("Extracting text from Word document...")
        doc = docx.Document(io.BytesIO(file_content))
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        logger.info(f"Extracted {len(text)} characters from document")
        return text
    except Exception as e:
        logger.error(f"Error extracting text from document: {e}")
        raise

def generate_audio_with_polly(text: str, destination: Path) -> None:
    """Generate audio from text using AWS Polly"""
    try:
        logger.info("Generating audio with AWS Polly...")
        polly_client = boto3.client('polly', region_name=APIConfig.AWS_REGION)
        
        response = polly_client.synthesize_speech(
            Text=text,
            OutputFormat='mp3',
            VoiceId='Joanna'
        )
        
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        with destination.open('wb') as audio_file:
            audio_file.write(response['AudioStream'].read())
        
        if not destination.exists():
            raise RuntimeError(f"Failed to generate audio with Polly")
        
        file_size = destination.stat().st_size
        logger.info(f"Generated audio: {file_size / 1024:.2f} KB")
        
    except Exception as e:
        logger.error(f"Error generating audio with Polly: {e}")
        raise

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
    document: UploadFile = File(..., description="Input Word document with text"),
    video_name: str = Form(..., description="Name of video file in S3 (e.g., reya.mp4)"),
    inference_steps: int = Form(20, description="Number of inference steps (default: 20)"),
    guidance_scale: float = Form(1.0, description="Guidance scale (default: 1.0)"),
    seed: int = Form(1247, description="Random seed (-1 for random)"),
    enable_deepcache: bool = Form(False, description="Enable DeepCache optimization")
):
    """
    Generate lip-synced video from S3 video and Word document text.
    Returns the processed video file directly for download.
    """
    request_id = str(uuid.uuid4())[:8]
    temp_files = []
    
    try:
        # ========================================================================
        # 1. Validate document input
        # ========================================================================
        
        if not document.filename.lower().endswith('.docx'):
            raise HTTPException(
                status_code=400,
                detail="Invalid document format. Only .docx files are allowed"
            )
        
        # ========================================================================
        # 2. Setup paths
        # ========================================================================
        
        base_dir = Path(APIConfig.TEMP_BASE_DIR).resolve()
        base_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = base_dir / f"{request_id}_{timestamp}_input.mp4"
        audio_path = base_dir / f"{request_id}_{timestamp}_audio.mp3"
        output_path = base_dir / f"{request_id}_{timestamp}_output.mp4"
        
        temp_files = [video_path, output_path]
        
        logger.info(f"[{request_id}] Video path: {video_path}")
        logger.info(f"[{request_id}] Audio path: {audio_path}")
        logger.info(f"[{request_id}] Output path: {output_path}")
        
        # ========================================================================
        # 3. Download video from S3
        # ========================================================================
        
        logger.info(f"[{request_id}] Downloading video from S3...")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            download_video_from_s3,
            APIConfig.S3_BUCKET_NAME,
            video_name,
            video_path
        )
        
        # Verify video exists
        if not video_path.exists():
            raise HTTPException(status_code=500, detail=f"Failed to download video from S3")
        
        # ========================================================================
        # 4. Extract text from document and generate audio with Polly
        # ========================================================================
        
        logger.info(f"[{request_id}] Processing Word document...")
        document_content = await document.read()
        
        text = await loop.run_in_executor(
            None,
            extract_text_from_docx,
            document_content
        )
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="Document contains no text")
        
        logger.info(f"[{request_id}] Generating audio with Polly...")
        await loop.run_in_executor(
            None,
            generate_audio_with_polly,
            text,
            audio_path
        )
        
        # Verify audio exists
        if not audio_path.exists():
            raise HTTPException(status_code=500, detail=f"Failed to generate audio with Polly")
        
        # ========================================================================
        # 5. Run inference
        # ========================================================================
        
        logger.info(f"[{request_id}] Starting inference...")
        
        def run_inference():
            engine.process(
                video_path=str(video_path),
                audio_path=str(audio_path),
                video_out_path=str(output_path),
                inference_steps=inference_steps,
                guidance_scale=guidance_scale,
                seed=seed,
                enable_deepcache=enable_deepcache,
                temp_dir="temp"
            )
        
        await loop.run_in_executor(None, run_inference)
        
        # ========================================================================
        # 6. Verify output exists
        # ========================================================================
        
        if not output_path.exists():
            raise HTTPException(
                status_code=500,
                detail="Inference completed but output file not found"
            )
        
        # ========================================================================
        # 7. Return file response
        # ========================================================================
        
        output_filename = f"lipsync_output_{timestamp}.mp4"
        
        logger.info(f"[{request_id}] ✅ Returning output file: {output_filename}")
        
        return FileResponse(
            path=str(output_path),
            media_type="video/mp4",
            filename=output_filename,
            background=None
        )
        
    except HTTPException:
        raise
        
    except Exception as e:
        logger.error(f"[{request_id}] ❌ Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
        
    finally:
        # Cleanup individual files after delay
        if temp_files:
            async def delayed_cleanup():
                await asyncio.sleep(300)  # 5 minutes
                for file_path in temp_files:
                    try:
                        if file_path.exists():
                            file_path.unlink()
                            logger.info(f"🧹 Deleted: {file_path.name}")
                    except Exception as e:
                        logger.warning(f"Cleanup failed for {file_path.name}: {e}")
            
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
        "workingmain:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Set to True for development
        log_level="info"
    )