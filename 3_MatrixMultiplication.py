# Question

"""
Matrix multiplication
There are many important types of matrices which have their uses in neural networks. Some important matrices are matrices of ones (where each entry is set to 1) and the identity matrix (where the diagonal is set to 1 while all other values are 0). The identity matrix is very important in linear algebra: any matrix multiplied with identity matrix is simply the original matrix.

Let us experiment with these two types of matrices. You are going to build a matrix of ones with shape 3 by 3 called tensor_of_ones and an identity matrix of the same shape, called identity_tensor. We are going to see what happens when we multiply these two matrices, and what happens if we do an element-wise multiplication of them.
"""

# Instructions

"""
Instructions
100 XP
Create a matrix of ones with shape 3 by 3, and put it on variable tensor_of_ones.
Create an identity matrix with shape 3 by 3, and put it on variable identity_tensor.
Do a matrix multiplication of tensor_of_ones with identity_tensor and print its value.
Do an element-wise multiplication of tensor_of_ones with identity_tensor and print its value.
"""
# Create a matrix of ones with shape 3 by 3
tensor_of_ones = torch.ones(3, 3)

# Create an identity matrix with shape 3 by 3
identity_tensor = torch.eye(3)

# Do a matrix multiplication of tensor_of_ones with identity_tensor
matrices_multiplied = torch.matmul(tensor_of_ones, identity_tensor)
print(matrices_multiplied)

# Do an element-wise multiplication of tensor_of_ones with identity_tensor
element_multiplication = tensor_of_ones * identity_tensor
print(element_multiplication)