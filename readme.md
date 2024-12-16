### Deep learning using pytorch

Deep learning is a subset of machine learning that uses neural networks to analyze data. PyTorch is  a popular deep learning framework that provides a dynamic computation graph and automatic differentiation. Here's an example of  how to use PyTorch for deep learning:  

`import torch
import torch.nn as nn

class MLP(nn.Module):
   def  __init__(self, input_dim, hidden_dim, output_dim):
      super(MLP,  self).__init__()
      self.fc1 = nn.Linear(input_dim, hidden_dim)
      self.fc2 = nn.Linear(hidden_dim, output_dim)

    def  forward(self, x):
       x = torch.relu(self.fc1(x))
       x = self.fc2(x)
       return x

`



