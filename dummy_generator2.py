import argparse
import numpy as np
import pandas as pd

def generate_profile_with_hamming_distance(reference_profile, num_changes, value_pool):
    profile = reference_profile.copy()
    indices_to_change = np.random.choice(len(reference_profile), num_changes, replace=False)
    for idx in indices_to_change:
        profile[idx] = np.random.choice(value_pool)
    return profile

def generate_dummy_dataset(num_samples, num_loci, max_hamming_distance_percentage, duplication_percentage, output_file):
    # Define the pool of possible allelic values including 0
    value_pool_size = num_loci * 10
    value_pool = np.arange(0, value_pool_size, dtype=np.int32)
    np.random.shuffle(value_pool)
    
    # Calculate the maximum number of changes allowed based on the percentage
    max_hamming_distance = int(num_loci * (max_hamming_distance_percentage / 100.0))
    
    # Calculate the number of duplicated samples
    num_duplicated_samples = int(num_samples * (duplication_percentage / 100.0))
    
    # Generate a random reference profile
    reference_profile = np.random.choice(value_pool, num_loci)
    
    # Generate sample names
    samples = [f'Sample{i+1}' for i in range(num_samples)]
    
    # Generate profiles with controlled Hamming distance
    data = np.zeros((num_samples, num_loci), dtype=np.int32)
    for i in range(num_samples - num_duplicated_samples):
        num_changes = np.random.randint(0, max_hamming_distance + 1)
        data[i] = generate_profile_with_hamming_distance(reference_profile, num_changes, value_pool)
    
    # Add duplicated samples by copying existing ones
    for i in range(num_samples - num_duplicated_samples, num_samples):
        original_index = np.random.randint(0, num_samples - num_duplicated_samples)
        data[i] = data[original_index]
    
    # Generate column names
    columns = ['ID'] + [f'L{i+1}.fasta' for i in range(num_loci)]
    
    # Combine sample names and data into a DataFrame
    df = pd.DataFrame(data, columns=columns[1:])
    df.insert(0, 'ID', samples)
    
    # Save to a file
    df.to_csv(output_file, sep='\t', index=False)
    print(f'Dummy dataset with {num_samples} samples, {num_loci} loci, max {max_hamming_distance_percentage}% Hamming distance, and {duplication_percentage}% duplicated samples saved to {output_file}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate a dummy dataset.')
    parser.add_argument('num_samples', type=int, help='Number of samples')
    parser.add_argument('num_loci', type=int, help='Number of loci')
    parser.add_argument('max_hamming_distance_percentage', type=int, help='Maximum percentage of Hamming distance between samples')
    parser.add_argument('duplication_percentage', type=int, help='Percentage of duplicated samples')
    parser.add_argument('output_file', type=str, help='Output file name')
    args = parser.parse_args()
    
    generate_dummy_dataset(args.num_samples, args.num_loci, args.max_hamming_distance_percentage, args.duplication_percentage, args.output_file)
