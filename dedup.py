import subprocess
import argparse
import os
import logging

def run_single_minhash_deduplication(
    base_path: str,
    data_file: str,         # Input: Specific filename
    output_path: str,       # Input: Specific output path
    split: str = "train",
    cache_dir: str = "./cache",
    column: str = "text",
    num_perm: int = 112,
    threshold: float = 0.75,
    b: int = 14,
    r: int = 8,
    batch_size: int = 10000,
):
    """
    Runs the text_dedup.minhash command for a single specified input file.

    Args:
        base_path (str): The base directory containing the data file.
        data_file (str): The specific data file name within the base_path
                         (e.g., "train-00001-of-00016.parquet").
        output_path (str): The full path where the output results should be saved.
        split (str): The dataset split to process (e.g., "train").
        cache_dir (str): Directory for caching intermediate results.
        column (str): The column containing text data in the Parquet file.
        num_perm (int): Number of permutations for MinHash.
        threshold (float): Jaccard similarity threshold for deduplication.
        b (int): Number of bands for LSH.
        r (int): Number of rows per band for LSH.
        batch_size (int): Processing batch size for text_dedup.
    """
    logging.info(f"Starting MinHash deduplication process for file: {os.path.join(base_path, data_file)}")
    logging.info(f"Output will be saved to: {output_path}")

    # Create cache directory if it doesn't exist
    if not os.path.exists(cache_dir):
        logging.info(f"Creating cache directory: {cache_dir}")
        os.makedirs(cache_dir, exist_ok=True) # exist_ok=True prevents error if dir exists

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        logging.info(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True) # exist_ok=True prevents error if dir exists

    # Construct the command as a list of arguments
    command = [
        "python", "-m", "text_dedup.minhash",
        "--path", base_path,
        "--split", split,
        "--data_files", data_file,  # Use the specific filename
        "--cache_dir", cache_dir,
        "--output", output_path,   # Use the specific output path
        "--column", column,
        "--num_perm", str(num_perm),
        "--threshold", str(threshold),
        "--b", str(b),
        "--r", str(r),
        "--batch_size", str(batch_size),
    ]

    logging.info(f"Executing command: {' '.join(command)}")

    try:
        # Execute the command
        process = subprocess.run(
            command,
            check=True,          # Raise exception on non-zero exit code
            capture_output=True, # Capture stdout/stderr
            text=True            # Decode stdout/stderr as text
        )
        logging.info(f"Successfully processed {data_file}.")
        logging.debug(f"stdout:\n{process.stdout}")
        if process.stderr:
             logging.debug(f"stderr:\n{process.stderr}")

    except subprocess.CalledProcessError as e:
        logging.error(f"Error processing file {data_file}:")
        logging.error(f"Command failed: {' '.join(e.cmd)}")
        logging.error(f"Return code: {e.returncode}")
        logging.error(f"stdout:\n{e.stdout}")
        logging.error(f"stderr:\n{e.stderr}")
        raise e # Reraise to indicate failure

    except FileNotFoundError:
        logging.error(f"Error: 'python' command not found or text_dedup not installed in the environment.")
        raise # Stop execution

    logging.info(f"MinHash deduplication finished for {data_file}.")


def main():
    parser = argparse.ArgumentParser(description='Run text_dedup.minhash for a single file.')

    # Required arguments for specifying the single file
    parser.add_argument('--base-path', type=str, required=True,
                        help='Base directory containing the data file (passed to text_dedup --path).')
    parser.add_argument('--data-file', type=str, required=True,
                        help='The specific input data file name (relative to base-path, e.g., "train-00001-of-00016.parquet").')
    parser.add_argument('--output-path', type=str, required=True,
                        help='The full path for the output results (e.g., "/content/drive/MyDrive/dedup/output_file_1.jsonl").')

    # Arguments corresponding to text_dedup parameters (with defaults)
    parser.add_argument('--split', type=str, default="train",
                        help='Dataset split to process.')
    parser.add_argument('--cache-dir', type=str, default="./cache",
                        help='Directory for caching intermediate results.')
    parser.add_argument('--column', type=str, default="text",
                        help='Column containing text data in the Parquet file.')
    parser.add_argument('--num-perm', type=int, default=112,
                        help='Number of permutations for MinHash.')
    parser.add_argument('--threshold', type=float, default=0.75,
                        help='Jaccard similarity threshold for deduplication.')
    parser.add_argument('--b', type=int, default=14,
                        help='Number of bands for LSH.')
    parser.add_argument('--r', type=int, default=8,
                        help='Number of rows per band for LSH.')
    parser.add_argument('--batch-size', type=int, default=10000,
                        help='Processing batch size for text_dedup.')

    args = parser.parse_args()

    # Call the main processing function with parsed arguments
    try:
        run_single_minhash_deduplication(
            base_path=args.base_path,
            data_file=args.data_file,
            output_path=args.output_path,
            split=args.split,
            cache_dir=args.cache_dir,
            column=args.column,
            num_perm=args.num_perm,
            threshold=args.threshold,
            b=args.b,
            r=args.r,
            batch_size=args.batch_size,
        )
        logging.info("Script finished successfully.")
    except Exception as e:
         logging.error(f"Script failed: {e}")
         # Exit with a non-zero status code to indicate failure
         exit(1)


if __name__ == '__main__':
    main()
