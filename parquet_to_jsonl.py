import pandas as pd
import json
import argparse
import sys
from pathlib import Path

def parquet_to_jsonl(input_file, output_file=None, chunk_size=10000):
    """
    Convert a Parquet file to JSONL format.
    
    Args:
        input_file (str): Path to input Parquet file
        output_file (str, optional): Path to output JSONL file. 
                                   If None, uses input filename with .jsonl extension
        chunk_size (int): Number of rows to process at once for memory efficiency
    """
    input_path = Path(input_file)
    
    if output_file is None:
        output_file = input_path.with_suffix('.jsonl')
    
    output_path = Path(output_file)
    
    print(f"Converting {input_path} to {output_path}")
    
    try:
        # Read the parquet file
        df = pd.read_parquet(input_path)
        
        print(f"Loaded DataFrame with shape: {df.shape}")
        
        # Convert to JSONL
        with open(output_path, 'w', encoding='utf-8') as f:
            # Process in chunks to handle large files efficiently
            total_rows = len(df)
            
            for start_idx in range(0, total_rows, chunk_size):
                end_idx = min(start_idx + chunk_size, total_rows)
                chunk = df.iloc[start_idx:end_idx]
                
                # Convert chunk to JSON lines
                for _, row in chunk.iterrows():
                    # Convert row to dict, then to JSON with proper UTF-8 handling
                    row_dict = row.to_dict()
                    json_line = json.dumps(row_dict, ensure_ascii=False, separators=(',', ':'))
                    f.write(json_line + '\n')
                
                print(f"Processed rows {start_idx + 1}-{end_idx} of {total_rows}")
        
        print(f"Successfully converted to {output_path}")
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
    except Exception as e:
        print(f"Error during conversion: {e}")

def main():
    parser = argparse.ArgumentParser(description="Convert Parquet file to JSONL format")
    parser.add_argument("input", help="Input Parquet file path")
    parser.add_argument("-o", "--output", help="Output JSONL file path (optional)")
    parser.add_argument("-c", "--chunk-size", type=int, default=10000,
                       help="Chunk size for processing (default: 10000)")
    
    args = parser.parse_args()
    
    parquet_to_jsonl(args.input, args.output, args.chunk_size)

if __name__ == "__main__":
    # Example usage when run directly
    if len(sys.argv) == 1:
        # Interactive mode - you can modify these paths
        input_file = "example.parquet"
        output_file = "example.jsonl"  # Optional, will auto-generate if None
        
        print("Running in example mode...")
        print("Modify the input_file and output_file variables in the script")
        print(f"Current input_file: {input_file}")
        print(f"Current output_file: {output_file}")
        
        # Uncomment the line below to run with example files
        # parquet_to_jsonl(input_file, output_file)
    else:
        main()