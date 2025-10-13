import os
import math
import json
import torch
import jsonlines
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForImageTextToText
from vsi_util import *


def inference_video(video, prompt, model, processor):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    return output_text[0]


def vsibench_aggregate_results(results):
    results_df = pd.DataFrame(results)

    output = {}

    for question_type, question_type_indexes in results_df.groupby('question_type').groups.items():
        per_question_type = results_df.iloc[question_type_indexes]
        
        if question_type in MCA_QUESTION_TYPES:
            for metric in METRICS_FOR_MCA.keys():
                output[f"{question_type}_{metric}"] = per_question_type[metric].mean()
        elif question_type in NA_QUESTION_TYPES:
            for metric in METRICS_FOR_NA.keys():
                output[f"{question_type}_{metric}"] = per_question_type[metric].mean()

        else:
            raise ValueError(f"Unknown question type: {question_type}")
    
    try:
        output['object_rel_direction_accuracy'] = sum([
            output.pop('object_rel_direction_easy_accuracy'),
            output.pop('object_rel_direction_medium_accuracy'),
            output.pop('object_rel_direction_hard_accuracy'),
        ]) / 3.
    except:
        output['object_rel_direction_accuracy'] =0
    output['overall_accuracy'] = sum([_ for _ in output.values()]) / len(output) 
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluation on VSI-bench for Qwen3-VL based models')
    parser.add_argument('--parquet_file', type=str, default='./VSI-Bench/test-00000-of-00001.parquet')
    parser.add_argument('--video_path', type=str, default='./VSI-Bench')
    parser.add_argument('--res_dir', type=str, default='./results')
    parser.add_argument('--save_name', type=str, default=None)
    parser.add_argument('--model', type=str, default='Qwen/Qwen3-VL-30B-A3B-Instruct')
    args = parser.parse_args()

    vsibench = pd.read_parquet(args.parquet_file)
    if args.save_name:
        res_path = os.path.join(args.res_dir, args.save_name)
    else:
        res_path = os.path.join(args.res_dir, args.model.split('/')[-1])
    
    os.makedirs(res_path, exist_ok=True)
    if os.path.exists(os.path.join(res_path, 'response.jsonl')):
        prev_response = list(jsonlines.open(os.path.join(args.res_path, 'response.jsonl')))
        prev_id = [item['id'] for item in prev_response]
    else:
        prev_id = []

    model = AutoModelForImageTextToText.from_pretrained(
        args.model, 
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(args.model)

    for i in tqdm(range(len(vsibench))):
        cur_data = vsibench.loc[i]
        if int(cur_data['id']) in prev_id:
            continue
        prompt = "These are frames of a video.\n" + cur_data['question']
        if cur_data['options'] is None:
            prompt += "\nPlease answer the question using a numerical value (e.g., 42 or 3.1)."
        else:
            options = cur_data['options'].tolist()
            prompt += "\nOptions:\n" + "\n".join(options)
            prompt += "\nAnswer with the option's letter from the given choices directly."

        video_url = os.path.join(args.video_path, f"{cur_data['dataset']}", f"{cur_data['scene_name']}.mp4")
        response = inference_video(video_url, prompt, model, processor)

        if cur_data['options'] is not None:
            if response in cur_data['options'].tolist():
                response = response.split('.')[0]

        with open(os.path.join(res_path, 'response.jsonl'), "a") as f:
            f.write(
                json.dumps(
                    {
                        "id": int(cur_data["id"]),
                        "predicted_answer": response,
                        'dataset': cur_data['dataset'],
                        'scene_name': cur_data['scene_name'],
                        'question': cur_data['question'],
                        'ground_truth': cur_data['ground_truth'],
                        'question_type': cur_data['question_type'],
                        'promt': prompt,
                    }
                ) + "\n"
            )

    # EVAL Results
    results = []
    with open(os.path.join(res_path, 'response.jsonl'), 'r') as f:
        for line in f:
            doc = json.loads(line)
            processed_doc = vsibench_process_results(doc)  # Process each doc to add metrics
            results.append(processed_doc)

    aggregated_results = vsibench_aggregate_results(results)

    with open(os.path.join(res_path, 'result.json'), "w") as f:
        json.dump(aggregated_results, f, indent=4, ensure_ascii=False)