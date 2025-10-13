# Evaluating Qwen3-VL on VSI-Bench

```shell
python evaluate \
    --parquet_file ./VSI-Bench/test-00000-of-00001.parquet \
    --video_path ./VSI-Bench \
    --res_dir ./results \
    --model Qwen/Qwen3-VL-30B-A3B-Instruct
```

## Results

```json
// Qwen3-VL-30B-A3B-Instruct

{
    "obj_appearance_order_accuracy": 0.7200647249190939,
    "object_abs_distance_MRA:.5:.95:.05": 0.536810551558753,
    "object_counting_MRA:.5:.95:.05": 0.7012367491166077,
    "object_rel_distance_accuracy": 0.6,
    "object_size_estimation_MRA:.5:.95:.05": 0.757607555089192,
    "room_size_estimation_MRA:.5:.95:.05": 0.6590277777777778,
    "route_planning_accuracy": 0.36082474226804123,
    "object_rel_direction_accuracy": 0.7266381807446698,
    "overall_accuracy": 0.6327762851842669
}
```