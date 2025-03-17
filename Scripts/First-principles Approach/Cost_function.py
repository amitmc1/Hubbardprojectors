import pandas as pd
import numpy as np


def calculate_j_fp(row, results):
    # Extract SISSO values from results
    sisso_p1_2 = results['SISSO_p1_2']
    sisso_p2_2 = results['SISSO_p2_2']
    sisso_p3_2 = results['SISSO_p3_2']

    # Extract pbe0 values from references
    pbe0_p1 = row['pbe0_p1']
    pbe0_p2 = row['pbe0_p2']
    pbe0_p3 = row['pbe0_p3']

    # Calculate percentage errors
    err_p1 = ((sisso_p1_2 - pbe0_p1) / pbe0_p1) * 100
    err_p2 = ((sisso_p2_2 - pbe0_p2) / pbe0_p2) * 100
    err_p3 = ((sisso_p3_2 - pbe0_p3) / pbe0_p3) * 100

    # Calculate the Euclidean norm of the percentage errors
    j_fp = np.sqrt(err_p1 ** 2 + err_p2 ** 2 + err_p3 ** 2)
    return j_fp


def process_references_and_results(reference_file, result_files_dir):
    # Read the references file
    references_df = pd.read_csv(reference_file, delimiter='\t')

    # Iterate over each row in the references dataframe (each support)
    for index, row in references_df.iterrows():
        support = row['Support']

        # Read the corresponding result file
        result_file = f"{result_files_dir}/{support}_results.txt"
        try:
            results_df = pd.read_csv(result_file, delimiter=',')
        except FileNotFoundError:
            print(f"Warning: Result file for {support} not found!")
            continue  # Skip if the result file doesn't exist

        # Extract the required SISSO columns (assuming single row per file for simplicity)
        results = results_df.iloc[0]  # First row (assuming only one row of results per file)

        # Prepare a list to store the new rows with J(FP) values
        new_rows = []

        # Iterate over each row in the result file and calculate J(FP)
        for i, result_row in results_df.iterrows():
            # Calculate the J(FP) value for each row
            j_fp = calculate_j_fp(row, result_row)
            new_row = result_row.tolist() + [j_fp]  # Add the J(FP) value to the row
            new_rows.append(new_row)

        # Create the header with an additional 'J(FP)' column
        header = list(results_df.columns) + ['J(FP)']

        # Write the new rows into a new file for this support
        output_file = f"{result_files_dir}/{support}_with_JFP.txt"
        try:
            with open(output_file, 'w') as f:
                f.write('\t'.join(header) + '\n')  # Write the header
                for row in new_rows:
                    f.write('\t'.join(map(str, row)) + '\n')  # Write the data rows
            print(f"Successfully wrote {support}_with_JFP.txt")
        except Exception as e:
            print(f"Error writing file for {support}: {e}")


# Define the paths to the references file and the directory containing the result files
reference_file = 'references.txt'
result_files_dir = r'C:\Users\Dell\PycharmProjects\Hubbardprojectors\Scripts\First-principles Approach'

# Process the files and create new result files with the J(FP) column
process_references_and_results(reference_file, result_files_dir)
