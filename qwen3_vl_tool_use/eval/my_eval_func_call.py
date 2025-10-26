import datasets


vsibench_dataset = datasets.load_dataset("parquet", data_files="/vepfs_c/gaolei/VSI-Bench/my_processed_data/train.parquet")['train']
# for sample in vsibench_dataset:

