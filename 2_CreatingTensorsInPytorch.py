# Question

"""
Creating tensors in PyTorch
Random tensors are very important in neural networks. Parameters of the neural networks typically are initialized with random weights (random tensors).

Let us start practicing building tensors in PyTorch library. As you know, tensors are arrays with an arbitrary number of dimensions, corresponding to NumPy's ndarrays. You are going to create a random tensor of sizes 3 by 3 and set it to variable your_first_tensor. Then, you will need to print it. Finally, calculate its size in variable tensor_size and print its value.

NB: In case you have trouble solving the problems, you can always refer to slides in the bottom right of the screen.

"""

# Instructions

"""
Instructions
100 XP
Import PyTorch main library.
Create the variable your_first_tensor and set it to a random torch tensor of size 3 by 3.
Calculate its shape (dimension sizes) and set it to variable tensor_size.
Print the values of your_first_tensor and tensor_size.

"""

# Import torch
import torch

# Create random tensor of size 3 by 3
your_first_tensor = torch.rand(3, 3)

# Calculate the shape of the tensor
tensor_size = your_first_tensor.size()

# Print the values of the tensor and its shape
print(your_first_tensor)
print(tensor_size)