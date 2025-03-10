import torch
import numpy as np



# Initialize tensor 
data = [[1,2],[3,4]]
x_data = torch.tensor(data)

# x_data의 속성은 유지
x_ones = torch.ones_like(x_data, dtype=torch.float)
print(x_ones)

x_rand = torch.rand_like(x_data, dtype=torch.float)
print(x_rand)

x_zero = torch.zeros_like(x_data)
print(x_zero)

shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}\n")




# Attribute of tensor 
""" 텐서의 속성은 텐서의 모양(shape), 자료형(datatype)및 어느 장치에 저장되는지를 나타낸다."""
tensor = torch.rand(3,4)
print(tensor)

print(f"Shape of tensor: {tensor.shape}\n")
print(f"Datatype of tensor: {tensor.dtype}\n")
print(f"Device tensor is stored on: {tensor.device}\n")



# Tensor Operation 
"""https://pytorch.org/docs/stable/torch.html
    1. 각 연산들은 GPU에서 실행할 수 있다.
    2. GPU availability를 확인한 뒤 GPU로 텐서를 명시적으로 이동할 수 있습니다.
    3. 기본적으로 tensor는 CPU에 저장된다.
    4. tensor의 indexing & slicing문법은 numpy와 동일하다.
"""

if torch.cuda.is_available():
    tensor = tensor.to("cuda")


tensor = torch.rand(3,4)
print(f"Origin tenosr: {tensor}\n")
print(f"First row: {tensor[0]}\n")
print(f"First_column: {tensor[:, 0]}\n")
print(f"Last Column: {tensor[..., -1]}\n")
tensor[:, 1] = 0
print(tensor)
print('\n\n')


print(f"tensor : {tensor}")
print('\n\n', '-'*20, "Concatenate Tensor(dim=1)", '-'*20)
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)
print(t1.shape)

print('\n\n', '-'*20, "Concatenate Tensor(dim=0)", '-'*20)
t2 = torch.cat([tensor, tensor, tensor], dim=0)
print(t2.shape)



print('\n\n', '-'*20, "Tensor Arithmetic operations", '-'*20)
"""matmul : matrix multiplication
tensor.T는 텐서의 전치를 반환한다."""
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)