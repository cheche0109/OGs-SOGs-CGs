import csv
import math
from Bio import SeqIO
from collections import Counter
from Bio.Data import IUPACData
import pandas as pd

# Define the Kyte-Doolittle hydropathy scale
kyte_doolittle_scale = {
    'A': 1.8,  'C': 2.5,  'D': -3.5, 'E': -3.5, 'F': 2.8,
    'G': -0.4, 'H': -3.2, 'I': 4.5,  'K': -3.9, 'L': 3.8,
    'M': 1.9,  'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
    'S': -0.8, 'T': -0.7, 'V': 4.2,  'W': -0.9, 'Y': -1.3
}

# Function for Shannon entropy
def calculate_shannon_entropy(sequence):
    counts = Counter(sequence)
    total_length = len(sequence)
    probabilities = [count / total_length for count in counts.values()]
    entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)
    return entropy

# Function for dinucleotide frequencies (10 features)
def dinucleotide_frequencies(sequence):
    dinucleotides = ['AT', 'TA', 'CG', 'AA', 'TT', 'CC', 'GG', 'AG', 'CT', 'GT', 'AC', 'GA', 'TC']
    counts = Counter(sequence[i:i+2] for i in range(len(sequence) - 1))
    total_dinucs = len(sequence) - 1
    features = {dinuc: counts.get(dinuc, 0) / total_dinucs if total_dinucs > 0 else 0 for dinuc in dinucleotides}
    return features

# Function for trinucleotide frequencies (64 features)
def calculate_trinucleotide_frequencies(sequence):
    all_trinucleotides = [a + b + c for a in "ATCG" for b in "ATCG" for c in "ATCG"]
    counts = Counter(sequence[i:i+3] for i in range(len(sequence) - 2))
    total_trinucs = sum(counts.values())
    features = {trinuc: counts.get(trinuc, 0) / total_trinucs if total_trinucs > 0 else 0 for trinuc in all_trinucleotides}
    return features

# Function for amino acid composition (20 features)
def calculate_aac(sequence):
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    counts = Counter(sequence)
    total_length = len(sequence)
    features = {aa: counts.get(aa, 0) / total_length if total_length > 0 else 0 for aa in amino_acids}
    return features

# Other feature calculations
def calculate_hydropathy_score(sequence, window_size=9, consistency_window=5):
    hydropathy_scores = [
        sum(kyte_doolittle_scale.get(aa, 0) for aa in sequence[i:i+window_size]) / window_size
        for i in range(len(sequence) - window_size + 1)
    ]
    consistent_count = sum(
        all(score > 0 for score in hydropathy_scores[i:i+consistency_window]) or
        all(score < 0 for score in hydropathy_scores[i:i+consistency_window])
        for i in range(len(hydropathy_scores) - consistency_window + 1)
    )
    return consistent_count / len(sequence)

def average_residue_weight(sequence):
    weights = IUPACData.protein_weights
    total_weight = sum(weights.get(aa, 0) for aa in sequence)
    return total_weight / len(sequence) if len(sequence) > 0 else 0

def average_charge(sequence, pH=7.0):
    pKa_values = {'D': 3.9, 'E': 4.2, 'R': 12.5, 'K': 10.5, 'H': 6.0}
    total_charge = sum(
        -1 / (1 + 10**(pH - pKa_values[aa])) if aa in 'DE' else
        1 / (1 + 10**(pKa_values[aa] - pH)) if aa in 'RKH' else 0
        for aa in sequence
    )
    return total_charge / len(sequence) if len(sequence) > 0 else 0

# Process sequences and output features
def process_sequences(dna_fasta, protein_fasta, output_csv):
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Header
        fieldnames = ['ID', 'DNA_Shannon_Entropy', 'Protein_Shannon_Entropy']
        fieldnames += [f'Dinuc_{key}' for key in ['AT', 'TA', 'CG', 'AA', 'TT', 'CC', 'GG', 'AG', 'CT', 'GT', 'AC', 'GA', 'TC']]
        fieldnames += [f'Trinuc_{trinuc}' for trinuc in [a + b + c for a in "ATCG" for b in "ATCG" for c in "ATCG"]]
        fieldnames += [f'AAC_{aa}' for aa in "ACDEFGHIKLMNPQRSTVWY"]
        fieldnames += ['Avg_Hydropathy', 'Avg_Residue_Weight', 'Avg_Charge']
        writer.writerow(fieldnames)

        # Set to track processed IDs to handle duplicates
        processed_ids = set()

        # Load DNA sequences, skipping duplicates
        dna_sequences = {}
        for record in SeqIO.parse(dna_fasta, "fasta"):
            accession = record.id
            if accession not in dna_sequences:
                dna_sequences[accession] = record

        # Load protein sequences and process them
        for record in SeqIO.parse(protein_fasta, "fasta"):
            accession = record.id

            # Skip if the ID has already been processed
            if accession in processed_ids:
                continue

            # Mark ID as processed
            processed_ids.add(accession)

            protein_seq = str(record.seq)
            dna_seq = str(dna_sequences.get(accession, "").seq)

            # Calculate features
            dna_shannon_entropy = calculate_shannon_entropy(dna_seq)
            protein_shannon_entropy = calculate_shannon_entropy(protein_seq)
            dinuc_features = dinucleotide_frequencies(dna_seq)
            trinuc_features = calculate_trinucleotide_frequencies(dna_seq)
            aac_features = calculate_aac(protein_seq)
            avg_hydropathy = calculate_hydropathy_score(protein_seq)
            avg_weight = average_residue_weight(protein_seq)
            avg_charge = average_charge(protein_seq)

            # Write row to CSV
            row = [accession, dna_shannon_entropy, protein_shannon_entropy]
            row += list(dinuc_features.values())
            row += list(trinuc_features.values())
            row += list(aac_features.values())
            row += [avg_hydropathy, avg_weight, avg_charge]
            writer.writerow(row)

# Run the script
process_sequences("../All_gene.fna", "../All_protein.faa", "All_Sequence.csv")
