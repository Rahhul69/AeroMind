import pandas as pd
import numpy as np

# 1. Load your original 500 rows
path_healthy = r"D:\Rahhhul\PBL\Dataset\healthy.csv"
path_faulty = r"D:\Rahhhul\PBL\Dataset\faulty.csv"

print("Loading original datasets...")
y_healthy = pd.to_numeric(pd.read_csv(path_healthy, sep=';', encoding='latin1', on_bad_lines='skip', decimal=',').iloc[:, 1], errors='coerce').dropna().values
y_faulty = pd.to_numeric(pd.read_excel(path_faulty, engine='openpyxl').iloc[:, -1], errors='coerce').dropna().values

# 2. The Augmentation Engine
def multiply_data(base_data, target_rows=100000, noise_factor=0.5):
    # Figure out how many times we need to loop the 500 rows to hit 100k
    repeats = int(np.ceil(target_rows / len(base_data)))
    repeated_data = np.tile(base_data, repeats)[:target_rows]
    
    # Generate random wind noise based on the standard deviation of your data
    noise = np.random.normal(0, noise_factor * np.std(base_data), target_rows)
    
    # Inject the noise into the looped data
    augmented_data = repeated_data + noise
    return augmented_data

# 3. Generate the massive new datasets
print("Augmenting healthy turbine data to 100,000 rows...")
healthy_100k = multiply_data(y_healthy)

print("Augmenting faulty turbine data to 100,000 rows...")
faulty_100k = multiply_data(y_faulty)

# 4. Save them as clean, normal CSVs
pd.DataFrame({'Amplitude': healthy_100k}).to_csv(r"D:\Rahhhul\PBL\Dataset\healthy_augmented.csv", index=False)
pd.DataFrame({'Amplitude': faulty_100k}).to_csv(r"D:\Rahhhul\PBL\Dataset\faulty_augmented.csv", index=False)

print("\nSuccess! Check your Dataset folder for the new augmented files.")