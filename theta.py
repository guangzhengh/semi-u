import pickle
import numpy as np
file_path = 'results\pairwise_rank_theta_data_20250618-184757.pkl'  # Replace with the actual path to your .pkl file

try:
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    print(f"Data successfully loaded from {file_path}:")
    print(data)
    supervise = []
    semi = []
    for key in data:
        if key['method'] == 'Supervised': #and key['model'] == 'linear':
            supervise.append(key['theta_hat'])
    supervise = np.array(supervise)
    print(supervise)
    print(np.std(supervise, axis=0))
    print(np.mean(supervise, axis=0))
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
except Exception as e:
    print(f"An error occurred while loading the pickle file: {e}")