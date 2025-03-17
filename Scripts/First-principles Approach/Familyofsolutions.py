import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Define the directory where the output files (with J(FP)) are stored
output_files_dir = r'C:\Users\Dell\PycharmProjects\Hubbardprojectors\Scripts\First-principles Approach'  # Adjust this path

# Define material groups
groups = {
    'Group 1': ['a-TiO2', 'r-TiO2', 'CeO2', 'Cu2O'],
    'Group 2': ['WO3', 'MoO3', 'ZrO2', 'Y2O3'],
    'Group 3': ['LiCoO2', 'LiFePO4']
}

# Create an empty DataFrame to store results
results = pd.DataFrame(columns=['Material', 'U', 'c1', 'c2', 'JFP'])


# Function to find the family of solutions for a given material
def find_family_of_solutions(material):
    global results
    output_file = f"{output_files_dir}/{material}_with_JFP.txt"

    if not os.path.exists(output_file):
        print(f"Warning: Output file for {material} not found!")
        return

    # Read the output file
    df = pd.read_csv(output_file, delimiter='\t')

    # Ensure the necessary columns exist
    if 'U' not in df.columns or 'c1' not in df.columns or 'c2' not in df.columns or 'J(FP)' not in df.columns:
        print(f"Error: Required columns not found in {material} file!")
        return

    # Group by U and find the c1, c2 that minimize J(FP)
    grouped = df.groupby('U').apply(lambda df: df.loc[df['J(FP)'].idxmin()])

    # Print out the family of solutions for each material
    print(f"Family of solutions for material {material}:")
    for idx, row in grouped.iterrows():
        print(f"Material={material}, U={row['U']}, c1={row['c1']}, c2={row['c2']}, J(FP)={row['J(FP)']}")

        # Create a DataFrame for the solution and append to results
        solution_df = pd.DataFrame({
            'Material': [material],
            'U': [row['U']],
            'c1': [row['c1']],
            'c2': [row['c2']],
            'JFP': [row['J(FP)']]
        })
        results = pd.concat([results, solution_df], ignore_index=True)


# Loop through all unique materials and find the family of solutions
unique_materials = sum(groups.values(), [])  # Flatten the list of materials
for material in unique_materials:
    find_family_of_solutions(material)

# Save results to a text file
results.to_csv('family_of_solutions_with_JFP.txt', sep='\t', index=False)
print("Family of solutions with J(FP) has been saved to 'family_of_solutions_with_JFP.txt'")

# Plotting the family of solutions for each group
n_groups = len(groups)
fig, axs = plt.subplots(n_groups, 2, figsize=(16, 5 * n_groups), constrained_layout=False)

# Define colors for each material
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_materials)))


# Define material labels with LaTeX formatting
def format_label(material):
    if material == 'a-TiO2':
        return r'a-TiO$_2$'
    elif material == 'r-TiO2':
        return r'r-TiO$_2$'
    elif material == 'Cu2O':
        return r'Cu$_2$O'
    elif material == 'CeO2':
        return r'CeO$_2$'
    elif material == 'WO3':
        return r'WO$_3$'
    elif material == 'MoO3':
        return r'MoO$_3$'
    elif material == 'ZrO2':
        return r'ZrO$_2$'
    elif material == 'Y2O3':
        return r'Y$_2$O$_3$'
    elif material == 'LiCoO2':
        return r'LiCoO$_2$'
    elif material == 'LiFePO4':
        return r'LiFePO$_4$'
    else:
        return material


# Collect handles and labels for the legend
handles = []
labels = []

# Plot each group
for i, (group_name, materials_list) in enumerate(groups.items()):
    for material in materials_list:
        if material in unique_materials:
            material_data = results[results['Material'] == material]
            color = colors[unique_materials.index(material)]  # Get color for the material

            # Plot U vs c1
            sc1 = axs[i, 0].scatter(material_data['U'], material_data['c1'], color=color, label=format_label(material))
            axs[i, 0].set_xlabel('Hubbard $U$ Value (eV)')
            axs[i, 0].set_ylabel('Hubbard Projector $c_{1}$')

            # Plot U vs c2
            sc2 = axs[i, 1].scatter(material_data['U'], material_data['c2'], color=color, label=format_label(material))
            axs[i, 1].set_xlabel('Hubbard $U$ Value (eV)')
            axs[i, 1].set_ylabel('Hubbard Projector $c_{2}$')

            # Collect handles and labels for the legend
            if len(handles) == 0 or format_label(material) not in labels:  # Only add new handles and labels
                handles.append(sc1)
                labels.append(format_label(material))

# Add a single horizontal legend above the entire figure
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1), ncol=len(labels) // 2)

# Adjust layout
plt.show()
