import csv
import torch
import random
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the fine-tuned model and tokenizer
model_path = "./bangla_qwen"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="cuda",
    torch_dtype=torch.float16,
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Function to generate response
def generate_response(prompt, max_length=400):
    formatted_prompt = f"### Instruction: {prompt} ### Response:"
    inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, padding=True).to("cuda")
    
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=max_length,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.6,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split("### Response:")[-1].split("###")[0].strip()
    return response

# Load dataset and select 20 questions
dataset = load_dataset("json", data_files="dataset\oporichita.json")["train"]
# Select 20 prompts (randomly if dataset has >20 examples, else take all available up to 20)
prompts = [example["prompt"] for example in dataset]
if len(prompts) > 20:
    prompts = random.sample(prompts, 20)
else:
    prompts = prompts[:20]
print(f"Selected {len(prompts)} prompts from oporichita.json")

# Generate responses and store results
results = []
for prompt in prompts:
    response = generate_response(prompt)
    results.append({"Prompt": prompt, "Response": response})
    print(f"\nPrompt: {prompt}")
    print(f"Response: {response}")

# Save results to CSV
output_file = "results\output.csv"
with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["Prompt", "Response"])
    writer.writeheader()
    writer.writerows(results)

print(f"\nResults saved to {output_file}")


