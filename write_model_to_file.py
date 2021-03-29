import torch
import io

filename: str = "models/model_best.pt"

torch.load(filename)

# Load ScriptModule from io.BytesIO object
with open(filename, 'rb') as f:
    buffer = io.BytesIO(f.read())

# Load all tensors to the original device
torch.load(buffer)

# Load all tensors onto CPU, using a device
buffer.seek(0)
torch.load(buffer, map_location=torch.device('cpu'))

# Load all tensors onto CPU, using a string
buffer.seek(0)
torch.load(buffer, map_location='cpu')

# Load with extra files.
extra_files = {'foo.txt': ''}  # values will be replaced with data
torch.load(filename, _extra_files=extra_files)
print(extra_files['foo.txt'])