

import torch
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
def generate_response(prompt, max_length=150):
    # Format prompt to match training
    formatted_prompt = f"### Instruction: {prompt} ### Response:"
    inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, padding=True).to("cuda")
    
    # Generate output
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=max_length,
        num_return_sequences=1,
        do_sample=True,  # Sampling for natural responses
        temperature=0.6,  # Moderate randomness
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id  # Ensure proper termination
    )
    
    # Decode and extract response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract the part after "Response:" and remove any trailing markers
    response = response.split("### Response:")[-1].split("###")[0].strip()
    return response

# Test the model
prompts = [
    "অনুপমের বাবা কী করে জীবিকা নির্বাহ করতেন?",
    "What was the profession of Anupam's father?",
    "What was the past of Shumbhunath’s family?",
    "শুম্ভুনাথবাবু কেন গহনা খুলে আনলেন?",
    "What trait is dominant in Anupam’s uncle’s character?",
    "Why didn’t Anupam protest at the wedding?",
    "অনুপমের বাবা কীভাবে অর্থ উপার্জন করেছিলেন?",
    "গল্পে কোন চরিত্র নারীর শক্তির প্রতীক?"


]
for prompt in prompts:
    print(f"\nPrompt: {prompt}")
    response = generate_response(prompt)
    print(f"Response: {response}")