from transformers import T5Tokenizer, AutoModelForSeq2SeqLM
import torch
import re
from Bio import SeqIO
from tqdm import tqdm  # For progress bar

# Set the device
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# Load the tokenizer and model
tokenizer = T5Tokenizer.from_pretrained('Rostlab/ProstT5', do_lower_case=False)
model = AutoModelForSeq2SeqLM.from_pretrained("Rostlab/ProstT5").to(device)
model.float() if device.type == 'cpu' else model.half()

# Define input and output file paths
input_fasta = "large_5.faa"  # Replace with your input FASTA file path
output_fasta = "3Di_conserved5_large.fasta"  # Replace with your desired output FASTA file path

# Batch size for processing sequences
batch_size = 1  # Adjust based on available GPU memory
print(batch_size)

# Function to process a batch of sequences
def translate_batch(sequences):
    # Replace ambiguous amino acids with 'X' and add white space between characters
    sequences = [" ".join(list(re.sub(r"[UZOB]", "X", seq))) for seq in sequences]
    sequences = ["<AA2fold> " + seq for seq in sequences]

    # Calculate min and max sequence lengths for the batch
    lengths = [len(seq.split()) for seq in sequences]
    min_len = min(lengths)
    max_len = max(lengths)

    # Tokenize and pad
    ids = tokenizer.batch_encode_plus(
        sequences,
        add_special_tokens=True,
        padding="longest",
        return_tensors='pt'
    ).to(device)

    # Generation configuration for "folding" (AA-->3Di)
    gen_kwargs_aa2fold = {
                      "do_sample": True,
                      "num_beams": 3, 
                      "top_p" : 0.95, 
                      "temperature" : 1.2, 
                      "top_k" : 6,
                      "repetition_penalty" : 1.2,
    }

    # translate from AA to 3Di (AA-->3Di)
    with torch.no_grad():
      translations = model.generate( 
                  ids.input_ids, 
                  attention_mask=ids.attention_mask, 
                  max_length=max_len, # max length of generated text
                  min_length=min_len, # minimum length of the generated text
                  early_stopping=True, # stop early if end-of-text token is generated
                  num_return_sequences=1, # return only a single sequence
                  **gen_kwargs_aa2fold
      )
    # Decode and remove white spaces
    decoded_translations = tokenizer.batch_decode(translations, skip_special_tokens=True)
    return ["".join(ts.split()) for ts in decoded_translations]

# Process the FASTA file in batches
with open(output_fasta, "w") as output_handle:
    batch = []
    headers = []
    total_sequences = sum(1 for _ in SeqIO.parse(input_fasta, "fasta"))  # Count total sequences

    print('Ready!') 
    with tqdm(total=total_sequences, desc="Processing Sequences", unit="seq") as pbar:
      for record in SeqIO.parse(input_fasta, "fasta"):
          batch.append(str(record.seq))
          headers.append(record.id)

          # Process the batch when it reaches the batch size
          if len(batch) == batch_size:
              translated_sequences = translate_batch(batch)
              for header, structure in zip(headers, translated_sequences):
                  output_handle.write(f">{header}\n{structure}\n")
              batch = []
              headers = []
              pbar.update(batch_size)

      # Process any remaining sequences
      if batch:
          translated_sequences = translate_batch(batch)
          for header, structure in zip(headers, translated_sequences):
              output_handle.write(f">{header}\n{structure}\n")
          pbar.update(len(batch)) 

print(f"Processing complete. Translated sequences saved to {output_fasta}")
