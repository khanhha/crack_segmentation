import torch

with open('models/model_best.pt', 'rb') as content_file:
    content = content_file.read()
with open('models/model_best.txt', 'w') as target_file:
    target_file.write(content)