import json
import os
from vsi_util import *

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


results = []
res_path = "/vepfs_c/gaolei/REVPT/eval_output/results_qwen3vl.jsonl"
a = 0
with open(res_path, 'r') as f:
    for line in f:
        a += 1
        doc = json.loads(line)
        processed_doc = vsibench_process_results(doc)  # Process each doc to add metrics
        results.append(processed_doc)
        # if a > 1000:
        #     break
aggregated_results = vsibench_aggregate_results(results)

result_path = "/vepfs_c/gaolei/REVPT/eval_output/metrics_new5.json"
with open(result_path, "w") as f:
    json.dump(aggregated_results, f, indent=4, ensure_ascii=False)