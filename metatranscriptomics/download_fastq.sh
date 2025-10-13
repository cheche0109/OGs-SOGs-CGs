#!/bin/bash
run_accession=$1  # Accepts the run accession as a command-line argument
# Create the output directory if it doesn't exist
OUTPUT_DIR="SRA_FastDump_Dir_test"
#mkdir -p $OUTPUT_DIR
# Download the FASTQ file(s) for the given run accession using fasterq-dump
fasterq-dump $run_accession -O $OUTPUT_DIR
# Compress the downloaded FASTQ files
if [ -f "$OUTPUT_DIR/${run_accession}_1.fastq" ] && [ -f "$OUTPUT_DIR/${run_accession}_2.fastq" ]; then
    gzip "$OUTPUT_DIR/${run_accession}_1.fastq"
    gzip "$OUTPUT_DIR/${run_accession}_2.fastq"
    echo "Paired-end FASTQ files downloaded and compressed: ${run_accession}_1.fastq.gz, ${run_accession}_2.fastq.gz"
elif [ -f "$OUTPUT_DIR/${run_accession}.fastq" ]; then
    gzip "$OUTPUT_DIR/${run_accession}.fastq"
    echo "Single-end FASTQ file downloaded and compressed: ${run_accession}.fastq.gz"
else
    echo "Error: No FASTQ files found for $run_accession"
    exit 1
fi