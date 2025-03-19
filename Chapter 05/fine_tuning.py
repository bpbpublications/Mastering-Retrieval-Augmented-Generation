# Import necessary libraries from Hugging Face and others
from huggingface_hub import notebook_login
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import math

# Load the SQuAD dataset from Hugging Face datasets
qa_dataset = load_dataset("squad")

# Set the model checkpoint to 'distilgpt2', a smaller GPT-2 model
model_name = 'distilgpt2'

# Load the tokenizer for the distilgpt2 model with the fast tokenizer implementation
text_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

# Fetch the special tokens (e.g., beginning-of-sequence token)
special_tokens_map = text_tokenizer.special_tokens_map

# Function to append a beginning-of-sequence token to the 'question' field in the dataset
def append_bos_to_question(entry):
    entry['question'] += special_tokens_map['bos_token']  # Append beginning-of-sequence token to the question
    return entry

# Remove unwanted columns from the dataset (keep only 'question')
qa_dataset = qa_dataset.remove_columns(['id', 'title', 'context', 'answers'])

# Apply the append_bos_to_question function to all questions in the dataset
qa_dataset = qa_dataset.map(append_bos_to_question)

# Tokenize the 'question' field using the tokenizer and truncate to a maximum length of 512 tokens
def tokenize_questions(batch):
    return text_tokenizer(batch['question'], truncation=True)

# Tokenize the dataset using the tokenize_questions function
tokenized_qa_dataset = qa_dataset.map(tokenize_questions, batched=True, num_proc=4, remove_columns=['question'])

# Set the maximum block length for tokenized text
max_seq_length = 128

# Function to divide the tokenized text into fixed-size blocks of 'max_seq_length' tokens
def create_token_blocks(tokenized_batch, seq_length):
    """
    Divides tokenized text into fixed-length blocks of size seq_length.
    """
    concatenated = {key: sum(tokenized_batch[key], []) for key in tokenized_batch.keys()}
    total_tokens = len(concatenated[list(tokenized_batch.keys())[0]])
    total_tokens = (total_tokens // seq_length) * seq_length  # Ensure total length is a multiple of block_size

    # Create blocks of tokenized text
    divided_batch = {
        key: [token_seq[i: i + seq_length] for i in range(0, total_tokens, seq_length)]
        for key, token_seq in concatenated.items()
    }

    divided_batch['labels'] = divided_batch['input_ids'].copy()  # Set the 'labels' to be identical to 'input_ids'
    return divided_batch

# Apply the block division function to the tokenized dataset
lm_prepared_dataset = tokenized_qa_dataset.map(
    lambda tokenized_batch: create_token_blocks(tokenized_batch, max_seq_length),
    batched=True,
    batch_size=1000,
    num_proc=4,
)

# Split the tokenized dataset into training and evaluation subsets
train_data = lm_prepared_dataset['train'].shuffle(seed=42).select(range(100))  # Use a subset of the training data
eval_data = lm_prepared_dataset['validation'].shuffle(seed=42).select(range(100))  # Use a subset of the evaluation data

# Load the distilgpt2 model for causal language modeling
text_generation_model = AutoModelForCausalLM.from_pretrained("distilgpt2")

# Add a padding token to the tokenizer (GPT-2 models do not have a padding token by default)
text_tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Define training arguments using Hugging Face's TrainingArguments class
train_params = TrainingArguments(
    output_dir=f'./{model_name}-squad',  # Where to save the results and checkpoints
    evaluation_strategy="epoch",  # Evaluate the model at the end of each epoch
    learning_rate=2e-5,  # Learning rate for optimization
    weight_decay=0.01,  # Weight decay for regularization
    push_to_hub=False,  # Set to True if you want to push the model to Hugging Face Hub
)

# Initialize the Trainer class to handle training, evaluation, and saving of the model
trainer = Trainer(
    model=text_generation_model,
    args=train_params,
    train_dataset=train_data,
    eval_dataset=eval_data,
    tokenizer=text_tokenizer,
)

# Evaluate the model and calculate perplexity (a measure of how well the model predicts the next token)
eval_result = trainer.evaluate()
print(f'Perplexity: {math.exp(eval_result["eval_loss"]):.2f}')

# Save the fine-tuned model and tokenizer
text_tokenizer.save_pretrained('distilgpt2-squad')
text_generation_model.save_pretrained('distilgpt2-squad')
print("Fine-tuning completed. Model and tokenizer saved.")
