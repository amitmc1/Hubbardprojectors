#This script is from the ChemRxiv pre-print "Machine learning generalised DFT+U projectors in a numerical atom-centred orbital framework" at the DOI 10.26434/chemrxiv-2025-332z0 DOI: 10.26434/chemrxiv-2025-332z0
#This script is used in Section 3.3.4, to perform a linear search of the first-principle cost function (J_FP) for all materials specified in References.txt
#The linear search of J_FP uses empirical correlations for DFT+U-predicted O 2p orbital occupancies computed using hierarchical symbolic regression, as defined in HI-SISSO.py

import csv
import subprocess


def extract_constants(file_path):
    constants = {}
    headers = []

    # Open and read the input file
    with open(file_path, mode='r') as file:
        reader = csv.reader(file, delimiter='\t')

        # Read the header row to get the constant names
        headers = next(reader)

        # Process each row in the file
        for row in reader:
            support_name = row[0]
            constants_for_support = {headers[i]: row[i] for i in range(1, len(row))}
            constants[support_name] = constants_for_support

    return headers, constants


def run_bruteforce_script(support_name, constants, script_path):
    # Construct the output filename based on the support_name or other logic
    output_filename = f"{support_name}_results.txt"

    # Prepare the command to run HI-SISSO.py with constants as arguments
    command = ['python', script_path]  # Start with the script path

    # Add constants as command-line arguments
    for name, value in constants.items():
        command.append(f'--{name.replace(" ", "_")}={value}')

    # Add the output filename as the last argument
    command.append(output_filename)

    # Print the command for debugging purposes
    print(f"Running command: {' '.join(command)}")

    # Execute the command
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        print(f"Output for {support_name}:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Error running script for {support_name}:\n{e.stderr}")


def main():
    file_path = 'References.txt'
    # Specify the full path for HI-SISSO.py
    script_path = r'C:\Users\Dell\PycharmProjects\Hubbardprojectors\Scripts\First-principles Approach\HI-SISSO.py'

    headers, constants = extract_constants(file_path)

    for support_name, const_dict in constants.items():
        print(f"Processing {support_name}...")
        run_bruteforce_script(support_name, const_dict, script_path)


if __name__ == "__main__":
    main()
