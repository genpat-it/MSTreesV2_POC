import argparse
import numpy as np
import pandas as pd

def generate_dummy_dataset(num_samples, num_columns, similarity, output_file):
    # Define a large enough pool of unique values to draw from
    value_pool_size = num_columns * 10
    value_pool = np.arange(10000, 1000000, dtype=np.int32)[:value_pool_size]
    np.random.shuffle(value_pool)
    
    # Divide the pool into non-overlapping sets for each locus
    predefined_sets = [value_pool[i*10:(i+1)*10] for i in range(num_columns)]
    
    # Generate column names
    columns = ['ID'] + [f'L{i+1}.fasta' for i in range(num_columns)]
    
    # Generate sample names
    samples = [f'Sample{i+1}' for i in range(num_samples)]
    
    # Calculate number of similar values per column based on similarity percentage
    num_similar_values = int(num_samples * (similarity / 100.0))
    
    # Generate random data for each column from the predefined sets
    data = np.zeros((num_samples, num_columns), dtype=np.int32)
    
    for j in range(num_columns):
        # Generate similar values
        similar_value = np.random.choice(predefined_sets[j])
        data[:num_similar_values, j] = similar_value
        
        # Generate the remaining unique values for the column
        remaining_values = np.random.choice(predefined_sets[j], num_samples - num_similar_values, replace=True)
        data[num_similar_values:, j] = remaining_values
    
    # Shuffle rows to mix similar and unique values
    np.random.shuffle(data)
    
    # Combine sample names and data into a DataFrame
    df = pd.DataFrame(data, columns=columns[1:])
    df.insert(0, 'ID', samples)
    
    # Save to a file
    df.to_csv(output_file, sep='\t', index=False)
    print(f'Dummy dataset with {num_samples} samples, {num_columns} columns, and {similarity}% similarity saved to {output_file}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate a dummy dataset.')
    parser.add_argument('num_samples', type=int, help='Number of samples')
    parser.add_argument('num_columns', type=int, help='Number of columns')
    parser.add_argument('similarity', type=int, help='Percentage of similarity between rows')
    parser.add_argument('output_file', type=str, help='Output file name')
    args = parser.parse_args()
    
    generate_dummy_dataset(args.num_samples, args.num_columns, args.similarity, args.output_file)
