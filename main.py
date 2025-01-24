from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import concatenate_datasets

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

print(dataset_2["train"][0])

#select only the columns needed (input and output)
dataset_1 = dataset_1.select_columns(["input", "output"])
dataset_2 = dataset_2.select_columns(["input", "output"])
dataset_3 = dataset_3.select_columns(["input", "output"])
dataset_4 = dataset_4.select_columns(["input", "output"])

#load tokenizer
tokenizer = AutoTokenizer.from_pretrained('gpt2')
#gpt2 has no padding token by default. so end of sequence token is introduced
tokenizer.pad_token = tokenizer.eos_token

#tokenize the dataset
def tokenize_function(examples):
  #concatenate the context and response
  text = [f"{input} {output}" for input, output in zip(examples['input'], examples['output'])]
  inputs = tokenizer(text, truncation = True, padding='max_length', max_length = 512)
  inputs['labels'] = inputs['input_ids'].copy()
  return inputs

# Applying the tokenize function to each dataset
tokenized_dataset_1 = dataset_1.map(tokenize_function, batched=True)
tokenized_dataset_2 = dataset_2.map(tokenize_function, batched=True)
tokenized_dataset_3 = dataset_3.map(tokenize_function, batched=True)
tokenized_dataset_4 = dataset_4.map(tokenize_function, batched=True)

#concatenatinating all the datasets
combined_tokenized_dataset = concatenate_datasets([tokenized_dataset_1["train"],
                                                   tokenized_dataset_2["train"],
                                                   tokenized_dataset_3["train"],
                                                   tokenized_dataset_4["train"]])




