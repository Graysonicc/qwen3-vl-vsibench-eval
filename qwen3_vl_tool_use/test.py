from datasets import load_dataset
from tqdm import tqdm
import polars as pl
import random
import os
from torch.utils.data import Subset

# vsi_bench = load_dataset("parquet", data_files="/vepfs_c/gaolei/VSI-Bench/my_processed_data/train.parquet")
vsi_bench = load_dataset("/vepfs_c/gaolei/VSI-Bench")
# print(vsi_bench)
# RL_PROMPT = """You are a specialized multimodal agent. Your purpose is to solve video question answering tasks by thinking step-by-step and using tools.

# # Tools

# You are provided with the following tools, which can operate on one or multiple video frames:
# 1. object_detection: Detects objects in one or multiple frames. parameters: frame_id (int or list of ints), objects (list of object names).
# 2. zoom_in: Zooms in on a specified bounding box in one or multiple frames. parameters: frame_id (int or list of ints), bbox (bounding box coordinates), factor (zoom factor).
# 3. edge_detection: Detects edges in one or multiple frames. parameters: frame_id (int or list of ints).
# 4. depth_estimation: Estimates depth in one or multiple frames. parameters: frame_id (int or list of ints).
# 5. bev_3d_reconstruction: Performs 3D reconstruction from one or multiple frames and returns a bird’s-eye view (BEV) image. parameters: frame_id (int or list of ints).

# # Instruction

# 1. The input is a video. Before using tools, you should reason about which frames (single or multiple) are most informative for the question — for example, key frames showing critical actions, objects, or transitions. You can specify frames by their index values (frame_id).
# 2. In each turn, start with a <think> tag. Within this tag, reason step-by-step about the video content and the question. Decide whether tool use is necessary and which frames to analyze. If tool results are already available, analyze them here as well.
# 3. If you decide to use tools, call them within a <tool_call> tag. For example:
#    <tool_call>
#    {"name": "object_detection", "arguments": {"frame_id": [3, 7, 12], "objects": ["car", "person"]}}
#    </tool_call>
#    This call applies object detection on frames 3, 7, and 12.
# 4. You may use multiple tools and multiple frame selections iteratively, depending on your reasoning.
# 5. Once you have gathered enough information, provide the final answer within an <answer> tag. Include a concise explanation of your reasoning and the final answer in \\boxed{} for verification.
# """

# "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"google_search-google_search\", \"description\": \"\\n    Searches Google using the Serper API and returns a formatted string with the results.\\n\\n    This tool requires a SERPER_API_KEY to be provided.\\n    The response is formatted as a Markdown string, summarizing the search results,\\n    which is suitable for consumption by large language models.\\n    \", \"parameters\": {\"type\": \"object\", \"properties\": {\"query\": {\"description\": \"The search query to ask Google.\", \"title\": \"Query\", \"type\": \"string\"}}, \"required\": [\"query\"]}}}\n{\"type\": \"function\", \"function\": {\"name\": \"image_describer-describe_image\", \"description\": \"\\n    Sends an image to the OpenAI API to get a detailed description.\\n    The image_path can be an absolute path or a path relative to the filesystem workspace.\\n    \", \"parameters\": {\"type\": \"object\", \"properties\": {\"image_path\": {\"description\": \"The path to the image file. Can be absolute or relative to the filesystem workspace.\", \"title\": \"Image Path\", \"type\": \"string\"}}, \"required\": [\"image_path\"]}}}\n{\"type\": \"function\", \"function\": {\"name\": \"file_loader-read_file\", \"description\": \"Reads the content of various file types and returns it as a string.\\n\\n    This tool intelligently handles different formats:\\n    - For plain text files (.txt, .py, .md, etc.), it can read a specific chunk of lines.\\n    - For documents like .docx, .pdf, .pptx, and .xlsx, it extracts and formats the text content.\\n    - For audio files (.mp3, .wav, etc.), it performs transcription to convert speech to text.\\n    - For XML files, it parses and formats the structure for readability.\\n    \", \"parameters\": {\"type\": \"object\", \"properties\": {\"file_path\": {\"description\": \"The absolute path to the file to be read.\", \"title\": \"File Path\", \"type\": \"string\"}, \"start_index\": {\"default\": 0, \"description\": \"For plain text files, the 0-based line number to start reading from. Defaults to 0.\", \"title\": \"Start Index\", \"type\": \"integer\"}, \"max_length\": {\"default\": 1000, \"description\": \"For plain text files, the maximum number of lines to read. Defaults to 1000.\", \"title\": \"Max Length\", \"type\": \"integer\"}}, \"required\": [\"file_path\"]}}}\n{\"type\": \"function\", \"function\": {\"name\": \"read-page-server-read_page\", \"description\": \"\\n    Read webpage content and generate AI-powered summaries based on user goals.\\n    \\n    This tool fetches webpage content using Jina API and then uses GPT to extract \\n    and summarize information that's relevant to the specified user goal.\\n    \", \"parameters\": {\"type\": \"object\", \"properties\": {\"url\": {\"description\": \"The URL(s) of the webpage(s) to visit. Can be a single URL string or a list of URLs.\", \"title\": \"Url\", \"type\": \"string\"}, \"goal\": {\"description\": \"The specific goal or objective for reading the webpage(s). This helps the AI focus on extracting relevant information.\", \"title\": \"Goal\", \"type\": \"string\"}}, \"required\": [\"url\", \"goal\"]}}}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>",


# {"type": "function", "function": {"name": "edge_detection", "description": "
#     Detects edges in one or multiple frames.
#     ", "parameters": {"type": "object", "properties": {
#         "frame_id": {"description": "List of frame indices to perform edge detection on.", "title": "Frame ID", "type": "array", "items": {"type": "integer"}}
#     }, "required": ["frame_id"]}}}
# {"type": "function", "function": {"name": "zoom_in", "description": "
#     Zooms in on specific regions of one or multiple frames.
#     ", "parameters": {"type": "object", "properties": {
#         "frame_id": {"description": "List of frame indices to zoom in on.", "title": "Frame ID", "type": "array", "items": {"type": "integer"}},
#         "bbox": {"description": "Bounding box coordinates for each region (list of [x1, y1, x2, y2]).", "title": "Bounding Box", "type": "array", "items": {"type": "array", "items": {"type": "integer"}}},
#         "factor": {"description": "Zoom-in scaling factor.", "title": "Factor", "type": "number"}
#     }, "required": ["frame_id", "bbox", "factor"]}}}


# QWEN3VL_PROMPT = """You are a specialized multimodal agent. Your purpose is to solve video question answering tasks by thinking step-by-step and using tools.
# # Tools

# You can choose whether to use tools to help you understand the video.

# You are provided with function signatures within <tools></tools> XML tags:

# <tools>
# {"type": "function", "function": {"name": "object_detection", "description": "
#     Detects objects in one or multiple frames.
#     ", "parameters": {"type": "object", "properties": {
#         "frame_id": {"description": "List of frame indices to process.", "title": "Frame ID", "type": "array", "items": {"type": "integer"}},
#         "objects": {"description": "List of object names to detect.", "title": "Objects", "type": "array", "items": {"type": "string"}}
#     }, "required": ["frame_id", "objects"]}}}
# {"type": "function", "function": {"name": "depth_estimation", "description": "
#     Estimates depth in one or multiple frames.
#     ", "parameters": {"type": "object", "properties": {
#         "frame_id": {"description": "List of frame indices to estimate depth for.", "title": "Frame ID", "type": "array", "items": {"type": "integer"}}
#     }, "required": ["frame_id"]}}}
# </tools>

# Before calling any tool, you must first think about your plan.

# For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
# <tool_call>
# {"name": <function-name>, "arguments": <args-json-object>}
# </tool_call>

# Once you have gathered enough evidence, wrap your reasoning and final answer in `<answer></answer>`.  
#    Include both:
#    - A concise reasoning summary, and  
#    - The final answer inside \\boxed{} for verification. For example, \\boxed{A}, \\boxed{12}, .etc.

# If you choose to use a tool:
#     - You can choose up to 8 frames in total.
#     - Each frame_id must be between 0 and 31 (inclusive).
#     - Do not include all frames; choose only the ones that are most useful or representative for the current task.
#     - Briefly explain why you selected these specific frames before using them.

# If you do not need to use any tools, you can directly answer the question according the video and put the final answer inside \\boxed{} for verification.
# """

QWEN3VL_PROMPT = """
# Tools

User can choose whether to use tools to help understand the video.

<tools>
{"type": "function", "function": {"name": "object_detection", "description": "
    Detects objects in one or multiple frames.
    ", "parameters": {"type": "object", "properties": {
        "frame_id": {"description": "List of frame indices to process.", "title": "Frame ID", "type": "array", "items": {"type": "integer"}},
        "objects": {"description": "List of object names to detect.", "title": "Objects", "type": "array", "items": {"type": "string"}}
    }, "required": ["frame_id", "objects"]}}}
{"type": "function", "function": {"name": "depth_estimation", "description": "
    Estimates depth in one or multiple frames.
    ", "parameters": {"type": "object", "properties": {
        "frame_id": {"description": "List of frame indices to estimate depth for.", "title": "Frame ID", "type": "array", "items": {"type": "integer"}}
    }, "required": ["frame_id"]}}}
{"type": "function", "function": {"name": "edge_detection", "description": "
    Detects edges in one or multiple frames.
    ", "parameters": {"type": "object", "properties": {
        "frame_id": {"description": "List of frame indices to perform edge detection on.", "title": "Frame ID", "type": "array", "items": {"type": "integer"}}
    }, "required": ["frame_id"]}}}
</tools>

For each function call, return a json object with the function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

If user chooses to use a tool:
    - If the target object user wants to observe appears in multiple frames, assistant should select the most suitable frame for tool call.
    - User can choose up to 8 frames in total.
After collecting the necessary information, assistant should wrap the final answer in \\boxed{} for verification.
"""
USER_PROMPT = """
If you choose to use a tool:
    - If the target object you want to observe appears in multiple frames, please select the most suitable frame for tool call.
    - You can choose up to 8 frames in total.
    - Do not include all frames; choose only the ones that are most useful or representative for the current task.
After collecting the necessary information, wrap your final answer in \\boxed{} for verification.
If you do not need to use any tools, you can directly answer the question based on the video and put the final answer inside \\boxed{} for verification, for example: \\boxed{A}, \\boxed{12}, etc.
"""


RL_PROMPT = """You are a specialized multimodal agent. Your purpose is to solve video question answering tasks by thinking step-by-step and using tools.

# Tools

You are provided with the following tools, which can operate on one or multiple video frames:
1. object_detection: Detects objects in one or multiple frames. 
   parameters: 
       - frame_id (list of ints)
       - objects (list of object names)

2. depth_estimation: Estimates depth in one or multiple frames. 
   parameters: 
       - frame_id (list of ints)

3. edge_detection: Detects edges in one or multiple frames.
   parameters:
       - frame_id (list of ints)

4. zoom_in: Zooms in on specific regions of one or multiple frames.
   parameters:
       - frame_id (list of ints)
       - bbox (bounding box coordinates, list of lists of 4 ints)
       - factor (float)
       
# Instruction

1. The input is a video. Before using tools, you should reason about which frames (single or multiple) are most informative for the question — for example, key frames showing critical actions, objects, or transitions. You can specify frames by their index values (frame_id).
2. In each turn, start with a <think> tag. Within this tag, reason step-by-step about the video content and the question. Decide whether tool use is necessary and which frames to analyze. If tool results are already available, analyze them here as well.
3. If you decide to use tools, call them within a <tool_call> tag. For example:
   <tool_call>
   {"name": "object_detection", "arguments": {"frame_id": list of ints, "objects": List of object names}}
   </tool_call>
   or
   <tool_call>
   {"name": "depth_estimation", "arguments": {"frame_id": list of ints}}
   </tool_call>
   or
   <tool_call>
   {"name": "edge_detection", "arguments": {"frame_id": list of ints}}
   </tool_call>
   or
   <tool_call>
   {"name": "zoom_in", "arguments": {"frame_id": list of ints, "bbox": list of lists of 4 ints, "factor": float}}
   </tool_call>
4. You may use multiple tools and multiple frame selections iteratively, depending on your reasoning.
5. Once you have gathered enough information, provide the final answer within an <answer> tag. Include a concise explanation of your reasoning and the final answer in \\boxed{} for verification.
6. Notice: "frame_id" must be a list, and each value must be between 0 and 31. You can select at most 5 frames at a time.
"""

TEXT_RL_PROMPT = "A conversation between User and Assistant." \
           "The user asks a question about the video, and the Assistant solves it." \
           "The assistant first thinks about the reasoning process and then provides the user with the answer. " \
           "The final answer must be enclosed within \\boxed{}, i.e., \\boxed{answer here}."


import cv2
def get_video_duration(video_path: str) -> float:
    """
    Get the duration of a video in seconds.
    
    Args:
        video_path (str): Path to the video file.
    
    Returns:
        float: Duration in seconds.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)          # 帧率
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)  # 总帧数
    cap.release()
    
    if fps == 0:
        raise ValueError("FPS is zero, cannot compute duration.")
    
    duration_sec = frame_count / fps
    return duration_sec


# Shuffle dataset indices
dataset = vsi_bench['test']

indices = list(range(len(dataset)))
random.seed(42)    
random.shuffle(indices)

shuffled_dataset = Subset(dataset, indices)


processed_data = []
for idx, row in tqdm(enumerate(shuffled_dataset), total=len(shuffled_dataset), desc=f"Processing test set"):
    if row['dataset'] == 'scannet':
        data_source = 'scannet'
        videos = ['file:///vepfs_c/gaolei/VSI-Bench/scannet/' + row['scene_name'] + '.mp4']
    elif row['dataset'] == 'arkitscenes':
        data_source = 'arkitscenes'
        videos = ['file:///vepfs_c/gaolei/VSI-Bench/arkitscenes/' + row['scene_name'] + '.mp4']
    elif row['dataset'] == 'scannetpp':
        data_source = 'scannetpp'
        videos = ['file:///vepfs_c/gaolei/VSI-Bench/scannetpp/' + row['scene_name'] + '.mp4']
    question = row['question']
    options = row["options"]
    if options:
        answer = row["ground_truth"]
        question += f"\nAnswer from the following choices:"
        for i, option in enumerate(options):
            question += f"\n{option}"
        question += '\nPut answer letter in the \\boxed{}'
    else:
        answer = row["ground_truth"]
        question += '\nPut answer numerical value in the \\boxed{}'

    # duration = get_video_duration(videos[0].split("file://")[-1])
    # print(duration)
    new_dict = {
        "data_source": data_source,
        "prompt": [
            {
                "role": "system",
                "content": TEXT_RL_PROMPT + QWEN3VL_PROMPT,
            },
            {
                "role": "user",
                "content": 'These are frames of a video.\n<video>\n' + question,
            }
        ],
        "videos": [{'video': video} for video in videos],
        "ability": "visual_spatial_reasoning",
        "reward_model": {"style": "rule", "ground_truth": answer},
        "extra_info": {
            "split": 'train',
            "index": idx, 
            "question": question,
        },
        "question_type": row['question_type']
    }
    processed_data.append(new_dict)

print(len(processed_data))

train_df = pl.from_dicts(processed_data)
local_dir = "/vepfs_c/gaolei/VSI-Bench/my_processed_data"
train_path = os.path.join(local_dir, "train.parquet")
print(f"Writing test set to {train_path}")
train_df.write_parquet(train_path)

print("Processing finished.")





# import pandas as pd

# def read_parquet_sample(file_path, sample_index):
#     """
#     读取 Parquet 文件并提取指定索引的样本。
    
#     参数:
#         file_path (str): Parquet 文件的路径。
#         sample_index (int): 要提取的样本索引。
    
#     返回:
#         pandas.Series: 指定索引的样本数据，如果索引超出范围，则返回 None。
#     """
#     try:
#         # 读取 Parquet 文件
#         df = pd.read_parquet(file_path)
        
#         # 检查索引是否在范围内
#         if sample_index < 0 or sample_index >= len(df):
#             print(f"Sample index {sample_index} is out of range. The file has {len(df)} rows.")
#             return None
        
#         # 提取指定索引的样本
#         sample = df.iloc[sample_index]
#         return sample
#     except Exception as e:
#         print(f"An error occurred: {e}")
#         return None

# # 示例用法
# file_path = "/vepfs_c/gaolei/VSI-Bench/my_processed_data/train.parquet"
# sample_index = 0  # 提取第 0 行的样本
# for i in range(5000):
#     sample = read_parquet_sample(file_path, i)
#     if sample is not None:
#         print("Sample data:")
#         print(sample)
