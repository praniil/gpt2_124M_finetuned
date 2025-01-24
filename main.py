from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

#used datasets
dataset_1 = load_dataset("marmikpandya/mental-health")
dataset_2 = load_dataset("RAJJ18/mental_health_dataset")
dataset_3 = load_dataset("fadodr/mental_health_therapy")
dataset_4 = load_dataset("Amod/mental_health_counseling_conversations")

#making the column name uniform
dataset_1 = dataset_1.rename_columns({
    "input": "input",
    "output": "output"
})

# Do the same for dataset_2 and dataset_3
dataset_2 = dataset_2.rename_columns({
    "input": "input",
    "output": "output"
})

dataset_3 = dataset_3.rename_columns({
    "input": "input",
    "output": "output"
})

dataset_4 = dataset_4.rename_columns({
    "Context": "input",
    "Response": "output"
})

