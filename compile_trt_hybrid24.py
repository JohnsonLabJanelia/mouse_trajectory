"""Compile TRT models for mouseHybrid24 predict2D pipeline."""
import sys, os, torch, torch_tensorrt

sys.path.insert(0, '/home/user/src/JARVIS-HybridNet')
from jarvis.config.project_manager import ProjectManager
from jarvis.efficienttrack.efficienttrack import EfficientTrack

trt_dir = '/home/user/src/JARVIS-HybridNet/projects/mouseHybrid24/trt-models/predict2D'
os.makedirs(trt_dir, exist_ok=True)

p = ProjectManager(); p.load('mouseHybrid24'); cfg = p.cfg
img_size_cd = cfg.CENTERDETECT.IMAGE_SIZE
bbox_size   = cfg.KEYPOINTDETECT.BOUNDING_BOX_SIZE

# ---- CenterDetect ----
print(f"Compiling CenterDetect ({img_size_cd}x{img_size_cd}) ...")
cd_model = EfficientTrack('CenterDetectInference', cfg, 'latest').model.eval().cuda()
traced = torch.jit.trace(cd_model, torch.randn(16, 3, img_size_cd, img_size_cd, device='cuda'))
trt_cd = torch_tensorrt.compile(
    traced, ir='ts',
    inputs=[torch_tensorrt.Input(
        min_shape=(1,  3, img_size_cd, img_size_cd),
        opt_shape=(16, 3, img_size_cd, img_size_cd),
        max_shape=(16, 3, img_size_cd, img_size_cd),
        dtype=torch.float)],
    enabled_precisions={torch.float16})
torch.jit.save(trt_cd, os.path.join(trt_dir, 'centerDetect.pt'))
print(f"  Saved centerDetect.pt")
del cd_model, traced, trt_cd; torch.cuda.empty_cache()

# ---- KeypointDetect ----
print(f"Compiling KeypointDetect ({bbox_size}x{bbox_size}) ...")
kd_model = EfficientTrack('KeypointDetectInference', cfg, 'latest').model.eval().cuda()
traced = torch.jit.trace(kd_model, torch.randn(8, 3, bbox_size, bbox_size, device='cuda'))
trt_kd = torch_tensorrt.compile(
    traced, ir='ts',
    inputs=[torch_tensorrt.Input(
        min_shape=(1,  3, bbox_size, bbox_size),
        opt_shape=(8,  3, bbox_size, bbox_size),
        max_shape=(16, 3, bbox_size, bbox_size),
        dtype=torch.float)],
    enabled_precisions={torch.float16})
torch.jit.save(trt_kd, os.path.join(trt_dir, 'keypointDetect.pt'))
print(f"  Saved keypointDetect.pt")
print("TRT compilation done!")
