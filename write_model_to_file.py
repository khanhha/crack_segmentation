import torch
import io

torch.jit.load('model_best.pt')

# Load ScriptModule from io.BytesIO object
with open('model_best.pt', 'rb') as f:
    buffer = io.BytesIO(f.read())

# Load all tensors to the original device
torch.jit.load(buffer)

# Load all tensors onto CPU, using a device
buffer.seek(0)
torch.jit.load(buffer, map_location=torch.device('cpu'))

# Load all tensors onto CPU, using a string
buffer.seek(0)
torch.jit.load(buffer, map_location='cpu')

# Load with extra files.
extra_files = {'foo.txt': ''}  # values will be replaced with data
torch.jit.load('model_best.pt', _extra_files=extra_files)
print(extra_files['foo.txt'])