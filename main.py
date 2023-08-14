from transformers import AutoTokenizer, AutoModelForCausalLM, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import os
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model = AutoModelForCausalLM.from_pretrained("gpt2")

path = r"/kaggle/input/datasets-squad-20/Dataset (3).txt"


cache_dir = "./cache_dir"
os.makedirs(cache_dir, exist_ok=True)  # Create the cache directory if it doesn't exist

text_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=path,
    block_size=128,
    cache_dir=cache_dir)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="./output",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_total_limit=2,
    save_steps=500,
    logging_dir="./logs",
    logging_steps=500,
    learning_rate=1e-4,
    warmup_steps=500,
    weight_decay=0.01,
)
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=text_dataset,
)
trainer.train()
model.save_pretrained("./fine_tuned_model")



from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "./fine_tuned_model"
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model = AutoModelForCausalLM.from_pretrained(model_path)

model.eval()

# Generate text using the model
prompt = """[Q] : In which decade did Beyonce become famous?
[A] :"""
input_ids = tokenizer.encode(prompt, return_tensors="pt")

# Generate text
output = model.generate(input_ids, max_length=50, num_return_sequences=1, no_repeat_ngram_size=2)

# Decode and print the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)