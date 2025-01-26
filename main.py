from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers import AutoConfig

model_name = "/home/pranil/gpt2_finetuned/results/model"

tokenizer = AutoTokenizer.from_pretrained(model_name)

config = AutoConfig.from_pretrained(model_name)

# Create a text generation pipeline
text_generator = pipeline("text-generation", model=model_name, tokenizer=tokenizer)

while True:
    user_input = input("You: ") 
    if user_input.lower() in ["exit", "quit"]:
        print("Bot: Goodbye!")
        break

    inputs = tokenizer(user_input, return_tensors='pt', padding=True, truncation=True)

    attention_mask = inputs['attention_mask']

    response = text_generator(
        user_input,
        max_length=200,  
        num_return_sequences=1,
        do_sample=True,
        temperature=0.7,
        truncation=False,
    )

    suggestion = response[0]['generated_text']
    suggestion = suggestion[len(user_input):].strip()  
    print(f"Bot: {suggestion}")
