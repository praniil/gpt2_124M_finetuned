from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import concatenate_datasets
import os
import torch

#used datasets
# dataset_2 = load_dataset("RAJJ18/mental_health_dataset")
dataset_1 = load_dataset("marmikpandya/mental-health")
dataset_3 = load_dataset("fadodr/mental_health_therapy")
dataset_4 = load_dataset("Amod/mental_health_counseling_conversations")

#making the column name uniform
dataset_1 = dataset_1.rename_columns({
    "input": "input",
    "output": "output"
})

# dataset_2 = dataset_2.rename_columns({
#     "input": "input",
#     "output": "output"
# })

dataset_3 = dataset_3.rename_columns({
    "input": "input",
    "output": "output"
})

dataset_4 = dataset_4.rename_columns({
    "Context": "input",
    "Response": "output"
})

# print(dataset_4["train"][0])

#select only the columns needed (input and output)
dataset_1 = dataset_1.select_columns(["input", "output"])
# dataset_2 = dataset_2.select_columns(["input", "output"])
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
#tokenized_dataset_2 = dataset_2.map(tokenize_function, batched=True)
tokenized_dataset_3 = dataset_3.map(tokenize_function, batched=True)
tokenized_dataset_4 = dataset_4.map(tokenize_function, batched=True)

#concatenatinating all the datasets
combined_tokenized_dataset = concatenate_datasets([tokenized_dataset_1["train"],
                                                #    tokenized_dataset_2["train"],
                                                   tokenized_dataset_3["train"],
                                                   tokenized_dataset_4["train"]])

print(len(combined_tokenized_dataset))

#finetuning the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = AutoModelForCausalLM.from_pretrained('gpt2')

#training argument define
training_args = TrainingArguments(
    output_dir = '/home/pranil/gpt2_finetuned/results',
    eval_strategy = 'epoch',
    num_train_epochs = 5,
    per_device_train_batch_size = 4,
    per_device_eval_batch_size = 4,
    warmup_steps = 100,
    weight_decay = 0.01,
    logging_dir = './logs',
    report_to = 'none'
)

#initialize trainer
trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = combined_tokenized_dataset,
    eval_dataset = tokenized_dataset_3["test"]
    # eval_dataset = combined_tokenized_dataset['train'].train_test_split(test_size = 0.2)
)

#train the model
trainer.train(resume_from_checkpoint='/home/pranil/gpt2_finetuned/results/checkpoint-16000')

model_output_dir = '/home/pranil/gpt2_finetuned/results/model'
#create the directory if it doesnot exist
os.makedirs(model_output_dir, exist_ok = True)

model.save_pretrained(model_output_dir)
tokenizer.save_pretrained(model_output_dir)




