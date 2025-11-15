import pandas as pd
import json
import argparse
import sys
from pathlib import Path

def jsonl_to_parquet(input_file, output_file=None, chunk_size=10000):
    """
    Convert a JSONL file to Parquet format.
    
    Args:
        input_file (str): Path to input JSONL file
        output_file (str, optional): Path to output Parquet file. 
                                   If None, uses input filename with .parquet extension
        chunk_size (int): Number of rows to process at once for memory efficiency
    """
    input_path = Path(input_file)
    
    if output_file is None:
        output_file = input_path.with_suffix('.parquet')
    
    output_path = Path(output_file)
    
    print(f"Converting {input_path} to {output_path}")
    
    try:
        # Read JSONL file in chunks
        chunks = []
        total_lines = 0
        
        with open(input_path, 'r', encoding='utf-8') as f:
            chunk_data = []
            
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                
                try:
                    row_dict = json.loads(line)
                    chunk_data.append(row_dict)
                    total_lines += 1
                    
                    # Process chunk when it reaches chunk_size
                    if len(chunk_data) >= chunk_size:
                        chunks.append(pd.DataFrame(chunk_data))
                        print(f"Processed {total_lines} lines...")
                        chunk_data = []
                        
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping malformed JSON at line {line_num}: {e}")
                    continue
            
            # Process remaining data
            if chunk_data:
                chunks.append(pd.DataFrame(chunk_data))
        
        if not chunks:
            print("Error: No valid data found in JSONL file.")
            return
        
        # Combine all chunks into a single DataFrame
        print(f"Combining {len(chunks)} chunks...")
        df = pd.concat(chunks, ignore_index=True)
        
        print(f"Created DataFrame with shape: {df.shape}")
        
        # Write to Parquet
        df.to_parquet(output_path, index=False, engine='pyarrow')
        
        print(f"Successfully converted to {output_path}")
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
    except Exception as e:
        print(f"Error during conversion: {e}")

def main():
    parser = argparse.ArgumentParser(description="Convert JSONL file to Parquet format")
    parser.add_argument("input", help="Input JSONL file path")
    parser.add_argument("-o", "--output", help="Output Parquet file path (optional)")
    parser.add_argument("-c", "--chunk-size", type=int, default=10000,
                       help="Chunk size for processing (default: 10000)")
    
    args = parser.parse_args()
    
    jsonl_to_parquet(args.input, args.output, args.chunk_size)

if __name__ == "__main__":
    # Example usage when run directly
    if len(sys.argv) == 1:
        # Interactive mode - you can modify these paths
        input_file = "example.jsonl"
        output_file = "example.parquet"  # Optional, will auto-generate if None
        
        print("Running in example mode...")
        print("Modify the input_file and output_file variables in the script")
        print(f"Current input_file: {input_file}")
        print(f"Current output_file: {output_file}")
        
        # Uncomment the line below to run with example files
        # jsonl_to_parquet(input_file, output_file)
    else:
        main()