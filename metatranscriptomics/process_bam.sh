#!/bin/bash

# Paths (edit as needed)
bam_dir="Bam_files"
output_dir="Coverage_output_final"

mkdir -p "$output_dir"

# Define a function that processes one BAM
process_bam() {
    bam_file="$1"
    sample_id=$(basename "$bam_file" .bam)
    output_txt="$output_dir/${sample_id}.txt"

    echo "🔄 Processing $sample_id"

    coverm contig \
        -b "$bam_file" \
        --min-read-percent-identity 95 \
        --min-read-aligned-percent 75 \
        --min-covered-fraction 0.6 \
        --contig-end-exclusion 0  \
        -o "$output_txt"

    echo "✅ Finished $sample_id"
}

export -f process_bam
export output_dir

# Run in parallel (adjust -j for number of jobs)
find "$bam_dir" -name "*.bam" | parallel -j 24 process_bam {}