import numpy as np
import requests
import io
from typing import List, Dict, Any, Optional, Tuple, Union
from PIL import Image
from qwen_agent.tools.base import BaseTool, register_tool
from qwen_agent.llm.fncall_prompts.nous_fncall_prompt import (
    NousFnCallPrompt,
    Message,
    ContentItem,
)
import cv2
import inspect
import sys
from PIL import ImageDraw
import random
import uuid
import os


def to_pil_image(frame: np.ndarray) -> Image.Image:
    """将 [3, H, W] 或 [H, W, 3] 的 numpy 数组安全转换为 PIL.Image"""
    # 如果是 [3, H, W] 格式，转成 [H, W, 3]
    if frame.ndim == 3 and frame.shape[0] == 3:
        frame = frame.transpose(1, 2, 0)

    # 判断 dtype 决定是否需要归一化
    if frame.dtype != np.uint8:
        # 通常说明是 [0,1] 的 float32，需要放缩回 [0,255]
        frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
    else:
        # 若原本就是 uint8，直接使用
        frame = np.ascontiguousarray(frame)

    return Image.fromarray(frame)

def fetch_tool_desc() -> str:
    # Get all classes in current module
    tool_classes = []
    current_module = sys.modules[__name__]
    
    for name, obj in inspect.getmembers(current_module):
        # Check if it's a class that inherits from BaseTool and has required attributes
        if (inspect.isclass(obj) and 
            issubclass(obj, BaseTool) and 
            obj != BaseTool):
            tool_classes.append(obj)
    
    tool_prompts = []
    for tool_class in tool_classes:
        tool_prompts.append(str(tool_class().function))
    
    return '\n'.join(tool_prompts)

def fetch_tools(placeholder='<|vision_start|><|image_pad|><|vision_end|>') -> List[BaseTool]:
    # Get all classes in current module
    tools = {}
    current_module = sys.modules[__name__]
    
    for name, obj in inspect.getmembers(current_module):
        # Check if it's a class that inherits from BaseTool and has required attributes
        if (inspect.isclass(obj) and 
            issubclass(obj, BaseTool) and 
            obj != BaseTool):
            tool_instance = obj(placeholder=placeholder)
            tools[tool_instance.name] = tool_instance
    
    return tools
    

class DepthEstimator(BaseTool):
    name = "depth_estimation"
    description = (
        "Depth estimation using the DepthAnything model. It returns the depth maps of one or multiple input frames. "
        "A 'Spectral_r' colormap is used to visualize depth — warmer colors indicate objects closer to the camera. "
        "This tool helps reason about spatial relationships in the video, such as which objects are closer or farther away."
    )

    parameters = [
        {
            "name": "frame_id",
            "type": "List[int]",
            "description": (
                "The list of frame indices from the input video that should be used for depth estimation. "
                "For example, [0, 5, 10] means to process frames 0, 5, and 10. "
                "Frame indices correspond to extracted video frames in the current conversation context, starting from 0."
            ),
            "required": True,
        }
    ]
    
    def __init__(self, url='http://localhost:9991/predict/color_depth', placeholder='<|vision_start|><|image_pad|><|vision_end|>'):
        self.url = url
        self.placeholder = placeholder
    
    def call(self, args: Dict[str, Any], env_state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            files = []
            # print("depth_estimation")
            # print(args['frame_id'])
            # print(env_state['video'])
            # print('---------------------------------------')
            if not isinstance(args['frame_id'], list):
                args['frame_id'] = [args['frame_id']]
            args['frame_id'] = [i for i in args['frame_id'] if int(i) < len(env_state['video'][0])]
            if len(args['frame_id']) == 0 or len(args['frame_id']) >= 32:
                args['frame_id'] = [0, 7, 15, 23, 31]
            print(args['frame_id'])
            print(len(env_state['video'][0]))
            print('---------------------------------------')
            if len(args['frame_id']) > 8:
                args['frame_id'] = sorted(random.sample(args['frame_id'], 8))

            for img_id in args['frame_id']:
                img_id = int(img_id)
                # env_state["video"][vid_id] 是 numpy 数组，形状 [n_frames, 3, H, W]
                video_np = env_state['video'][0][img_id]
                # 仅当 video_np 是 numpy 数组时才进行转换
                if isinstance(video_np, np.ndarray):
                    image = to_pil_image(video_np)
                else:
                    image = video_np
                if image.mode == 'RGBA':
                    image = image.convert('RGB')
                img_bytes = io.BytesIO()
                image.save(img_bytes, format='JPEG')
                img_bytes = img_bytes.getvalue()
                files.append(('files', (f'image_{img_id}.jpeg', img_bytes, 'image/jpeg')))

            # 1. 发送请求
            response = requests.post(self.url, files=files)
            response.raise_for_status()

            # 2. 获取 JSON 返回结果
            result_data = response.json()
            image_paths = result_data["results"]  # 一组图片路径
            # print(image_paths)
            # print('---------------------------------------')
            # 3. 依次读取这些图片
            returned_images = []
            for path in image_paths:
                with open(path, "rb") as f:
                    img = Image.open(io.BytesIO(f.read()))
                    returned_images.append(img)

            # 4. 组装返回结果
            return {
                "text": f"The colored depth maps for frames {args['frame_id']}.",
                "image": returned_images,
                "frame_ids": args['frame_id']
            }

        except KeyError as e:
            return {
                "text": (
                    f"Failed to detect objects for frame(s) {args['frame_id']} "
                    f"due to a key error: {str(e)}. Please check the tool_call format."
                ),
                "image": None
            }

        except Exception as e:
            return {
                "text": (
                    f"Failed to generate depth map(s) for frame(s) {args['frame_id']} "
                    f"due to error: {str(e)}"
                ),
                "image": None
            }



class Reconstruction_3d(BaseTool):
    name = "bev_3d_reconstruction"
    description = "3D reconstruction using the VGGT model. It takes the input image and performs 3D reconstruction, " \
                "then returns a bird’s-eye view (BEV) image of the reconstructed scene. " \
                "This tool may help you better understand the spatial layout of objects in the scene, " \
                "such as their positions and relative distances from a top-down perspective."
    parameters = [
        {
            'name': 'image_id',
            'type': 'int',
            'description': "The ID of the input image in the conversation (starting from 0). "
                        "This image will be used for 3D reconstruction, and the output will be "
                        "the bird’s-eye view (BEV) image of the reconstructed scene.",
            'required': True,
        }
    ]
    def __init__(self, url='http://localhost:9995/predict/recon_3d', placeholder='<|vision_start|><|image_pad|><|vision_end|>'):
        self.url = url
        self.placeholder = placeholder

    def call(self, args: Dict[str, Any], env_state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            image = env_state['image'][args['image_id']]
                        
            # Save PIL Image object to in-memory byte stream
            img_byte_arr = io.BytesIO()
            
            # Handle RGBA mode images, convert to RGB to avoid save errors
            if image.mode == 'RGBA':
                image = image.convert('RGB')
                
            image.save(img_byte_arr, format='JPEG')  # Change format if required by model
            img_byte_arr = img_byte_arr.getvalue()

            # Send POST request, upload image data as file
            files = {'file': ('image.jpeg', img_byte_arr, 'image/jpeg')}
            response = requests.post(self.url, files=files)
            response.raise_for_status()  # Raise exception if not 200 OK

            # Read image data from response and create PIL Image object
            returned_image = Image.open(io.BytesIO(response.content))

            return {
                "text": f"The colored bird’s-eye view (BEV) image of the reconstructed scene for image {args['image_id']}. ",
                "image": [returned_image]
            }
        except KeyError as e:
            return {
                "text": f"Failed to output the bird’s-eye view (BEV) image of the reconstructed scene for image {args['image_id']} due to error: Key error: {str(e)}. Please check the tool_call format.",
                "image": None
            }
        except Exception as e:
            return {
                "text": f"Failed to generate the bird’s-eye view (BEV) image of the reconstructed scene for image {args['image_id']} due to error: {str(e)}",
                "image": None
            } 

            
class EdgeDetector(BaseTool):
    name = "edge_detection"
    description = (
        "Uses Scharr edge detection to emphasize object contours in video frames. "
        "This tool helps identify boundaries and shapes in videos, by processing selected frames."
    )
            
    parameters = [
        {
            "name": "frame_id",
            "type": "List[int]",
            "description": (
                "The list of frame indices from the input video to process for edge detection. "
                "For example, [0, 5, 10] means to process frames 0, 5, and 10. "
                "Frame indices correspond to extracted video frames in the current context, starting from 0."
            ),
            "required": True,
        }
    ]
    
    def __init__(self, placeholder='<|vision_start|><|image_pad|><|vision_end|>'):
        self.placeholder = placeholder
    
    def call(self, args: Dict[str, Any], env_state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if not isinstance(args['frame_id'], list):
                args['frame_id'] = [args['frame_id']]
            args['frame_id'] = [i for i in args['frame_id'] if int(i) < len(env_state['video'][0])]
            if len(args['frame_id']) == 0 or len(args['frame_id']) >= 32:
                args['frame_id'] = [0, 7, 15, 23, 31]
            print(args['frame_id'])
            print(len(env_state['video'][0]))
            print('---------------------------------------')
            if len(args["frame_id"]) > 8:
                args["frame_id"] = sorted(random.sample(args["frame_id"], 8))

            returned_images=[]

            for fid in args['frame_id']:
                fid = int(fid)
                # 从 env_state['video'][0] 取出对应帧（PIL.Image）
                image = env_state["video"][0][fid]
                if not isinstance(image, Image.Image):
                    raise ValueError(f"Frame {fid} is not a PIL image.")
                # 转 numpy 数组
                img_np = np.array(image)
                # 转灰度
                if len(img_np.shape) == 3:
                    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                else:
                    gray = img_np
                # Scharr 边缘检测
                grad_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
                grad_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
                magnitude = cv2.magnitude(grad_x, grad_y)

                # 归一化到0-255
                normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

                # 转回PIL
                edge_img = Image.fromarray(normalized)
                returned_images.append(edge_img)

            return {
                "text": f"Generated Scharr edge maps for frames {args['frame_id']}.",
                "image": returned_images,
                "frame_ids": args["frame_id"]
            }
        except KeyError as e:
            return {
                "text": f"Failed to detect edges for frame(s) {args.get('frame_id', '?')} due to missing key: {str(e)}.",
                "image": None
            }

        except Exception as e:
            return {
                "text": f"Failed to generate edge map(s) for frame(s) {args.get('frame_id', '?')} due to error: {str(e)}",
                "image": None
            }
            
# class Segmentation(BaseTool):
#     name = "segmentation"
#     description = "Use a segmentation model to segment the image, and add colorful masks on the segmented objects. " \
#                   "Mode can be 'auto' (default, segment the entire image) or 'point' (segment the region around a specific point). " \
#                   "DO NOT use this tool to search or detect an object. Better after zooming in the image. " \
            
#     parameters = [
#         {
#             'name': 'image_id',
#             'type': 'int',
#             'description': "The ID of the image in the conversation including images from tool start from 0.",
#             'required': True,
#         },
#         {
#             'name': 'mode',
#             'type': 'str',
#             'description': "Segmentation mode: 'auto' (segment the entire image) or 'point' (segment based on a specific point).",
#             'required': False,
#             'default': 'auto',
#         },
#         {
#             'name': 'point',
#             'type': 'list',
#             'description': "When mode is 'point', specify the [x, y] coordinates of the point to segment around. Only needed with point mode.",
#             'required': False,
#         }
#     ]
    
#     def __init__(self, url='http://localhost:9992/predict/segmentation', placeholder='<|vision_start|><|image_pad|><|vision_end|>'):
#         self.url = url
#         self.placeholder = placeholder
    
#     def call(self, args: Dict[str, Any], env_state: Dict[str, Any]) -> Dict[str, Any]:
#         try:
#             image = env_state['image'][args['image_id']]
            
#             mode = args.get('mode', 'auto')
#             point = args.get('point', None)
            
#             if mode not in ['auto', 'point']:
#                 return {
#                     "text": f"Invalid segmentation mode: {mode}. Must be 'auto' or 'point'.",
#                     "image": None
#                 }
                
#             if mode == 'point' and point is None:
#                 return {
#                     "text": f"Point mode requires 'point' parameter with [x, y] coordinates.",
#                     "image": None
#                 }
                        
#             img_byte_arr = io.BytesIO()
            
#             if image.mode == 'RGBA':
#                 image = image.convert('RGB')
                
#             image.save(img_byte_arr, format='JPEG')  
#             img_byte_arr = img_byte_arr.getvalue()

#             files = {'file': ('image.jpeg', img_byte_arr, 'image/jpeg')}
            
#             form_data = {'mode': mode}
            
#             if mode == 'point' and point:
#                 form_data['point'] = f"{point[0]},{point[1]}"
                
#             response = requests.post(self.url, files=files, data=form_data)
#             response.raise_for_status()  
#             returned_image = Image.open(io.BytesIO(response.content))

#             if mode == 'auto':
#                 response_text = f"The segmentation mask for image {args['image_id']}."
#             else:
#                 response_text = f"The segmentation mask for image {args['image_id']} around point {point}."

#             return {
#                 "text": response_text,
#                 "image": [returned_image]
#             }
#         except KeyError as e:
#             return {
#                 "text": f"Failed to segment image {args['image_id']} due to error: Key error: {str(e)}. Please check the tool_call format.",
#                 "image": None
#             }
#         except Exception as e:
#             return {
#                 "text": f"Failed to generate segmentation mask for image {args['image_id']} due to error: {str(e)}",
#                 "image": None
#             }
            
class ZoomIn(BaseTool):
    name = "zoom_in"
    description = (
        "Enlarges specific regions in video frames to highlight intricate details, "
        "helping inspect small objects or regions in videos."
    )
    parameters = [
        {
            "name": "frame_id",
            "type": "List[int]",
            "description": (
                "The list of frame indices from the input video to zoom in on. "
                "For example, [0, 5, 10] means to process frames 0, 5, and 10. "
                "Frame indices correspond to extracted video frames in the current context, starting from 0."
            ),
            "required": True,
        },
        {
            'name': 'bbox',
            'type': 'list',
            'description': (
                "Bounding boxes for zoom-in regions. "
                "Can be a single bbox [x1, y1, x2, y2] or a list of per-frame bboxes. "
                "If a single bbox is given, it will be applied to all frames."
            ),
            'required': True,
        },
        {
            "name": "factor",
            "type": "float",
            "description": "The magnification factor. Between 1.0 and 2.0. Default 1.0.",
            "required": False,
            "default": 1.0,
        }
    ]
    
    def __init__(self, placeholder='<|vision_start|><|image_pad|><|vision_end|>'):
        self.placeholder = placeholder
    
    def call(self, args: Dict[str, Any], env_state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            video_frames = env_state['video'][0]  # list of PIL images
            frame_ids = args["frame_id"]
            if not isinstance(frame_ids, list):
                frame_ids = [frame_ids]
            frame_ids = [i for i in frame_ids if int(i) < len(env_state['video'][0])]
            if len(frame_ids) == 0 or frame_ids == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]:
                frame_ids = [0, 7, 15, 23, 31]
            bboxes = args['bbox']
            magnification = args.get('factor', 1.0)
            magnification = min(2.0, max(1.0, magnification))  # clamp
            if not isinstance(video_frames, list):
                raise ValueError("Video must be a list of PIL frames.")

            # 支持单个bbox或逐帧bbox
            if len(bboxes) == 4 and all(isinstance(i, (int, float)) for i in bboxes):
                bboxes = [bboxes] * len(frame_ids)
            elif len(bboxes) != len(frame_ids):
                raise ValueError("Number of bboxes must match number of frame_id if multiple bboxes are provided.")
            
            if len(frame_ids) > 8:
                sampled_indices = sorted(random.sample(range(len(frame_ids)), 8))  # 随机抽5个索引并排序
                frame_ids = [frame_ids[i] for i in sampled_indices]
                bboxes = [bboxes[i] for i in sampled_indices]

            zoomed_frames = []
            for idx, fid in enumerate(frame_ids):
                if fid < 0 or fid >= len(video_frames):
                    print(f"[Warning] frame_id {fid} is out of range, skipped.")
                    continue
                frame = video_frames[fid]
                bbox = bboxes[idx]

                img_w, img_h = frame.size
                x1, y1, x2, y2 = bbox

                # 限制 bbox 在图像范围内
                x1 = max(0, min(img_w, x1))
                y1 = max(0, min(img_h, y1))
                x2 = max(0, min(img_w, x2))
                y2 = max(0, min(img_h, y2))
                if x2 <= x1 or y2 <= y1:
                    print(f"[Warning] invalid bbox {bbox} at frame {fid}, skipped.")
                    continue

                # 裁剪 + 放大
                cropped = frame.crop((x1, y1, x2, y2))
                new_w = int(cropped.width * magnification)
                new_h = int(cropped.height * magnification)
                zoomed = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)

                zoomed_frames.append(zoomed)
            
                # ===================================================
                save_dir = "/vepfs_c/gaolei/REVPT/visualization/zoom"
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"zoomed_{uuid.uuid4().hex}.jpg")
                zoomed.save(save_path)
                # ===================================================

            if not zoomed_frames:
                return {"text": "No valid frames were zoomed.", "image": None}
            return {
                "text": f"Zoomed-in Frame {frame_ids} with {magnification}x magnification.",
                "image": zoomed_frames,
            }

        except KeyError as e:
            return {
                "text": f"Failed to detect object for Frame {frame_ids} due to error: Key error: {str(e)}. Please check the tool_call format.",
                "image": None
            }
        except Exception as e:
            return {
                "text": f"Failed to zoom in on Frame {frame_ids} due to error: {str(e)}",
                "image": None
            }


# class Transpose(BaseTool):
#     name = "transpose"
#     description = "Transforms images by flipping or rotating them to aid in viewing from different angles. "
            
#     parameters = [
#         {
#             'name': 'image_id',
#             'type': 'int',
#             'description': "The ID of the image in the conversation including images from tool start from 0.",
#             'required': True,
#         },
#         {
#             'name': 'operation',
#             'type': 'str',
#             'description': "The operation to perform. Options: 'ROTATE_90', 'ROTATE_180', 'ROTATE_270', 'FLIP_LEFT_RIGHT', 'FLIP_TOP_BOTTOM'.",
#             'required': True,
#         }
#     ]
    
#     def __init__(self, placeholder='<|vision_start|><|image_pad|><|vision_end|>'):
#         self.placeholder = placeholder
    
#     def call(self, args: Dict[str, Any], env_state: Dict[str, Any]) -> Dict[str, Any]:
#         try:
#             image = env_state['image'][args['image_id']]
            
#             type2id = {
#                 'FLIP_LEFT_RIGHT': 0,
#                 'FLIP_TOP_BOTTOM': 1,
#                 'ROTATE_90': 2,
#                 'ROTATE_180': 3,
#                 'ROTATE_270': 4
#             }
#             # Transpose the image
#             transposed_image = image.transpose(method=Image.Transpose(type2id[args['operation']]))
            
#             return {
#                 "text": f"The transposed image {args['image_id']}. ",
#                 "image": [transposed_image]
#             }
#         except KeyError as e:
#             return {
#                 "text": f"Failed to detect object for image {args['image_id']} due to error: Key error: {str(e)}. Please check the tool_call format.",
#                 "image": None
#             }
#         except Exception as e:
#             return {
#                 "text": f"Failed to transpose image {args['image_id']} due to error: {str(e)}",
#                 "image": None
#             }
            
class ObjectDetection(BaseTool):
    name = "object_detection"
    # description = "Object detection using Grounding DINO model. It returns the annotated image and the bounding boxes of the detected objects. " \
    #               "The detector is not perfect , it may wrongly detect objects or miss some objects. You should use the output as a reference , not as a ground truth. Better after zooming in the image."
    description = (
        "Object detection using Grounding DINO model. It returns annotated images "
        "and bounding boxes for detected objects across multiple frames. "
        "The detector is not perfect; results are for reference only."
        "Note: 'frame_id' refers to the index after frame sampling."
    )

    parameters = [
        {
            'name': 'frame_id',
            'type': 'List[int]',
            'description': (
                "The IDs of the frames in the video to detect objects from, starting from 0."
                "Frame indices correspond to extracted video frames in the current conversation context, starting from 0."
            ),
            'required': True,
        },
        {
            'name': 'objects',
            'type': 'List[str]',
            'description': "The objects to detect in the frames.",
            'required': True,
        }
    ]
    
    def __init__(self, url='http://localhost:9993/detect', placeholder='<|vision_start|><|image_pad|><|vision_end|>'):
        self.url = url
        self.placeholder = placeholder
    
    def call(self, args: Dict[str, Any], env_state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            files = []
            # print("object_detection")
            # print(args['frame_id'])
            # print(env_state['video'])
            # print('---------------------------------------')
            if not isinstance(args['frame_id'], list):
                args['frame_id'] = [args['frame_id']]
            args['frame_id'] = [i for i in args['frame_id'] if int(i) < len(env_state['video'][0])]
            if len(args['frame_id']) == 0 or len(args['frame_id']) >= 32:
                args['frame_id'] = [0, 7, 15, 23, 31]
            print(args['frame_id'])
            print(len(env_state['video'][0]))
            print('---------------------------------------')
            if len(args['frame_id']) > 8:
                args['frame_id'] = sorted(random.sample(args['frame_id'], 8))

            for img_id in args['frame_id']:
                img_id = int(img_id)
                # env_state["video"][vid_id] 是 numpy 数组，形状 [n_frames, 3, H, W]
                video_np = env_state['video'][0][img_id]
                # 仅当 video_np 是 numpy 数组时才进行转换
                if isinstance(video_np, np.ndarray):
                    image = to_pil_image(video_np)
                else:
                    image = video_np
                if image.mode == 'RGBA':
                    image = image.convert('RGB')
                img_bytes = io.BytesIO()
                image.save(img_bytes, format='JPEG')
                img_bytes = img_bytes.getvalue()
                files.append(('files', (f'image_{img_id}.jpeg', img_bytes, 'image/jpeg')))

            # Send POST request, upload image data as file
            objects = (". ".join(args['objects'])+".").lower()
            
            data = {
                "text_prompt": objects,
                "box_threshold": 0.2,
                "text_threshold": 0.3,
            }
            response = requests.post(self.url, files=files, data=data)
            response.raise_for_status()
            
            if response.headers.get('content-type') == 'application/json':
                json_response = response.json()

                frame_results = json_response.get("results", [])
                images_out = []
                detection_text = ""
                for res in frame_results:
                    if "message" in res:
                        detection_text += f"No objects matching '{objects}' detected in Frame {res.get('frame_id', '?')}\n."
                        continue

                    fid = res.get("frame_id", "?")
                    detected = res.get("detected_objects", [])
                    detection_text += f"Frame {fid}: Detected {len(detected)} object(s):\n"
                    for obj in detected:
                        detection_text += f" - {obj['label']} ({obj['confidence']:.2f}) @ {[int(x) for x in obj['bbox']]}\n"
                    
                    vis_path = res.get("visualization_path")
                    if vis_path:
                        try:
                            images_out.append(Image.open(vis_path))
                        except Exception:
                            pass
                return {
                    "text": detection_text,
                    "image": images_out if images_out else None,
                    "frame_ids": args['frame_id']
                }
            
        except KeyError as e:
            return {
                "text": f"Failed to detect objects due to missing key: {str(e)}. Please check the tool_call format.",
                "image": None
            }

        except Exception as e:
            return {
                "text": f"Failed to detect objects due to error: {str(e)}",
                "image": None
            }

        #         if "message" in json_response:
        #             return {
        #                 "text": f"No objects matching '{objects}' detected in image {args['image_id']}." \
        #                        "The result of the detection may be wrong, don't treat it as ground truth.",
        #                 "image": None
        #             }
                    
        #         detected_objects = json_response.get('detected_objects', [])
        #         detection_text = f"Detected {len(detected_objects)} object(s) in image {args['image_id']}:\n"
                
        #         for obj in detected_objects:
        #             detection_text += f"{obj['id']}. {obj['label']}({obj['confidence']:.2f}): {[int(x) for x in obj['bbox']]}\n"
        #         detection_text += f"Detection result may be wrong, don't treat it as ground truth."
                    
        #         visualization_path = json_response.get('visualization_path')
        #         result_image = None
                
        #         if visualization_path:
        #             try:
        #                 result_image = Image.open(visualization_path)
        #                 return {
        #                     "text": f"{detection_text}",
        #                     "image": [result_image]
        #                 }
        #             except Exception as e:
        #                 return {
        #                     "text": f"{detection_text}\nFailed to load visualization: {str(e)}",
        #                     "image": None
        #                 }
        #         else:
        #             return {
        #                 "text": detection_text,
        #                 "image": None
        #             }
                    
        #     elif response.headers.get('content-type', '').startswith('image/'):
        #         result_image = Image.open(io.BytesIO(response.content))
        #         return {
        #             "text": f"Objects detection for image {args['image_id']} ",
        #             "image": [result_image]
        #         }
                
        #     else:
        #         return {
        #             "text": f"Unexpected response type for image {args['image_id']}. Cannot process result.",
        #             "image": None
        #         }
        # except KeyError as e:
        #     return {
        #         "text": f"Failed to detect object for image {args['image_id']} due to error: Key error: {str(e)}. Please check the tool_call format.",
        #         "image": None
        #     }
        # except Exception as e:
        #     return {
        #         "text": f"Failed to detect object for image {args['image_id']} due to error: {str(e)}",
        #         "image": None
        #     }