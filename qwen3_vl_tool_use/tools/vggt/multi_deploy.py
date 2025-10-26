import io
import cv2
import torch.multiprocessing as mp
import torch
import numpy as np
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_preprocess_image
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response, FileResponse, JSONResponse
from contextlib import asynccontextmanager
import time
import queue
import tempfile
from PIL import Image

mp.set_start_method('spawn', force=True)

worker_pool = []
worker_index = 0

@asynccontextmanager
async def lifespan(app: FastAPI):
    global worker_pool, worker_index
    
    # Detect available GPU count
    gpu_count = torch.cuda.device_count()
    if gpu_count == 0:
        raise RuntimeError("No GPU devices available")
    print(f"Detected {gpu_count} GPU devices")

    for gpu_id in range(gpu_count):
        worker = ModelWorker(gpu_id=gpu_id)
        worker.start()
        worker_pool.append(worker)

    print(f"Initialized model worker processes on {len(worker_pool)} GPUs")
    
    yield

    print("Shutting down all model worker processes")
    for worker in worker_pool:
        worker.stop()
    worker_pool = []

app = FastAPI(
    title="VGGT API",
    description="API for VGGT 3D construction",
    version="1.0.0",
    lifespan=lifespan
)

def point_cloud_to_bev_pca(points, colors, img_size=1024, save_path=None):
    """
    使用 PCA 对齐房间主方向生成 BEV 彩色图。
    
    points: (N,3) np.array, 点云
    colors: (N,3) np.uint8, 点颜色
    img_size: 输出图片大小
    save_path: 保存路径
    
    return: bev_img (img_size,img_size,3)
    """
    points_xy = points[:, :2]

    # PCA: 计算主轴
    mean_xy = np.mean(points_xy, axis=0)
    centered = points_xy - mean_xy
    cov = centered.T @ centered
    eigvals, eigvecs = np.linalg.eigh(cov)
    rot_matrix = eigvecs[:, ::-1]  # 最大特征值对应方向

    # 旋转点云，使主方向水平
    points_rot = centered @ rot_matrix
    if np.mean(points_rot[:,0]) < 0:
        points_rot[:,0] *= -1  # 翻转 X 轴

    # 归一化到图像坐标
    x, y = points_rot[:,0], points_rot[:,1]
    x_norm = ((x - x.min()) / (x.max() - x.min()) * (img_size - 1)).astype(np.int32)
    y_norm = ((y - y.min()) / (y.max() - y.min()) * (img_size - 1)).astype(np.int32)

    bev_img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    bev_img[img_size - 1 - y_norm, x_norm] = colors  # 注意Y轴翻转

    return bev_img


class ModelWorker:
    def __init__(self, gpu_id: int):
        self.gpu_id = gpu_id
        self.device = f'cuda:{gpu_id}'
        self.request_queue = mp.Queue()
        self.result_queue = mp.Queue()
        self.process = None

    def start(self):
        """Start model worker process"""
        self.process = mp.Process(
            target=self._worker_process, 
            args=(self.gpu_id, self.request_queue, self.result_queue)
        )
        self.process.daemon = True
        self.process.start()

    def _worker_process(self, gpu_id: int, request_queue: mp.Queue, result_queue: mp.Queue):
        """Worker process function, loads model and handles requests"""
        try:
            torch.cuda.set_device(gpu_id)

            model = VGGT()
            local_path = "/vepfs_c/gaolei/vggt/model/model.pt"
            state_dict = torch.load(local_path, map_location=f"cuda:{gpu_id}") 
            model.load_state_dict(state_dict)
            model = model.to(f"cuda:{gpu_id}").eval()
            print(f"Model loaded to GPU:{gpu_id}")

            # Add counter, clean up memory every N requests
            request_count = 0
            clean_interval = 10  # Clean up every 10 requests

            while True:
                try:
                    # Get request
                    request_id, image = request_queue.get()
                    images_pre = load_preprocess_image(image).to(f"cuda:{gpu_id}")  # (1, 3, H, W)

                    print("Running inference...")
                    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
                    with torch.no_grad():
                        with torch.cuda.amp.autocast(dtype=dtype):
                            pred_dict = model(images_pre)

                    print("Converting pose encoding to extrinsic and intrinsic matrices...")
                    extrinsic, intrinsic = pose_encoding_to_extri_intri(pred_dict["pose_enc"], images_pre.shape[-2:])
                    pred_dict["extrinsic"] = extrinsic
                    pred_dict["intrinsic"] = intrinsic

                    print("Processing model outputs...")
                    for key in pred_dict.keys():
                        if isinstance(pred_dict[key], torch.Tensor):
                            pred_dict[key] = pred_dict[key].cpu().numpy().squeeze(0)  # remove batch dimension and convert to numpy

                    print("Visualizing 3D points by unprojecting depth map by cameras")
                    print("Starting viser visualization...")

                    images = pred_dict["images"]
                    depth_map = pred_dict["depth"]  # (S, H, W, 1)
                    extrinsics_cam = pred_dict["extrinsic"]  # (S, 3, 4)
                    intrinsics_cam = pred_dict["intrinsic"]  # (S, 3, 3)
                    world_points = unproject_depth_map_to_point_map(depth_map, extrinsics_cam, intrinsics_cam)
                    colors = images.transpose(0, 2, 3, 1)  # now (S, H, W, 3)
                    points = world_points.reshape(-1, 3)
                    colors_flat = (colors.reshape(-1, 3) * 255).astype(np.uint8)
                    scene_center = np.mean(points, axis=0)
                    points_centered = points - scene_center
                    bev_img = point_cloud_to_bev_pca(points_centered, colors_flat, save_path="bev_color.png")
                    result = {"colored_bev": bev_img}
                    result_queue.put((request_id, result))
                    # Clean up memory
                    request_count += 1
                    if request_count % clean_interval == 0:
                        torch.cuda.empty_cache()

                except Exception as e:
                    print(f"GPU {gpu_id} worker process error: {str(e)}")
                    result_queue.put((request_id, {"error": str(e)}))

        except Exception as e:
            print(f"GPU {gpu_id} initialization failed: {str(e)}")
    
    def process_image(self, image) -> dict:
        request_id = f"{time.time()}_{id(image)}"
        self.request_queue.put((request_id, image))

        # Wait for result
        while True:
            try:
                result_id, result = self.result_queue.get(timeout=30)  # 30 seconds timeout
                if result_id == request_id:
                    return result
            except queue.Empty:
                raise TimeoutError("Request processing timed out")

    def stop(self):
        """Stop worker process"""
        if self.process and self.process.is_alive():
            self.process.terminate()
            self.process.join()

def get_next_worker():
    """Simple round-robin scheduling to get the next worker process"""
    global worker_index, worker_pool
    
    if not worker_pool:
        raise RuntimeError("No available model worker processes")
    
    worker = worker_pool[worker_index]
    worker_index = (worker_index + 1) % len(worker_pool)
    return worker

@app.post("/predict/recon_3d")
async def predict_3d_construction(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")
    
    try:
        # Read uploaded image
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents))
        # Convert to RGB to ensure correct format
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        # Get worker and process image
        worker = get_next_worker()
        result = worker.process_image(pil_image)
        
        if "error" in result:
            raise Exception(result["error"])
        
        colored_bev = result["colored_bev"]
        colored_bev_pil = Image.fromarray(colored_bev)

        # Save and return depth map
        with tempfile.NamedTemporaryFile(suffix=".png", dir="/vepfs_c/gaolei/REVPT/visualization/color_bev", delete=False) as tmp:
            colored_bev_pil.save(tmp.name)
            print('-------ok--------\n')
            return FileResponse(tmp.name, media_type="image/png", filename="bev.png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")