# Question

"""
Forward pass
Let's have something resembling more a neural network. The computational graph has been given below. You are going to initialize 3 large random tensors, and then do the operations as given in the computational graph. The final operation is the mean of the tensor, given by torch.mean(your_tensor).
"""

# Instructions

"""
Instructions
100 XP
Initialize random tensors x, y and z, each having shape (1000, 1000).
Multiply x with y, putting the result in tensor q.
Do an elementwise multiplication of tensor z with tensor q, putting the results in f
"""
# Initialize tensors x, y and z
x = torch.rand(1000, 1000)
y = torch.rand(1000, 1000)
z = torch.rand(1000, 1000)

# Multiply x with y
q = torch.matmul(x,y)

# Multiply elementwise z with q
f = z * q

mean_f = torch.mean(f)
print(mean_f)