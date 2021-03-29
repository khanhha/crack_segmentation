import torch
import json
import sys

original_stdout = sys.stdout # Save a reference to the original standard output

with open('models/model_best.txt', 'w') as target_file:
    sys.stdout = target_file # Change the standard output to the file we created.
    print(torch.load('models/model_best.pt',  map_location=torch.device('cpu')))
    sys.stdout = original_stdout # Reset the standard output to its original value


    # target_file.write(torch.load('models/model_best.pt',  map_location=torch.device('cpu')))