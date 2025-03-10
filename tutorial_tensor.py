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
print(f'{y1}\n{y2}\n{y3}')

"""요소별 곱을 계산"""
print('\n'*3)
shape = (4,4)
tensor = torch.ones_like(tensor, dtype=float)
tensor[...,1] = 0

"""@는 행렬곱(.matmul), *는 요소별 곱(element-wise product)"""
test1 = tensor @ tensor.T
test2 = tensor * tensor 
print(f'{test1}\n{test2}')

tensor = torch.rand_like(tensor, dtype=float)
print(tensor)
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))


"""바꿔치기(in-place)연산 / pandas in-place연산 생각하면 쉬움"""
print('\n\n', '-'*20, "in-place연산", '-'*20)
print(f"{tensor}\n")
tensor.add_(5)
print(tensor)




"""Numpy변환 (Bridge)
이때 CPU상의 tensor와 NumPy배열은 메모리 공간을 공유하기 때문에, 하나를 변경하면 다른 하나도 변경된다."""
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")
print(f"t type : {type(t)} \n n type : {type(n)}")

t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

"""Numpy 배열을 텐서로 변환하기"""
n = np.ones(5)
t = torch.from_numpy(n)

np.add(n, 1, out=n)
print(f"t : {t}")
print(f"n : {n}")