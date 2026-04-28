from wildlife_datasets import datasets
import pandas as pd
import os


dataset_name = "DogFaceNet"
data_dir = f"data/{dataset_name}"
output_csv = f"{data_dir}/custom_labels.csv" 

datasets.__dict__[dataset_name].get_data(data_dir)

dataset = datasets.__dict__[dataset_name](data_dir)

print("Dataset summary:")
print(dataset.summary)
print("\nFirst few entries:")
print(dataset.df.head())

