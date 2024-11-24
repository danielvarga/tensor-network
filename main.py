import torch
import torch.nn as nn


class TensorNetworkBase(nn.Module):
    def __init__(self, d, k):
        super(TensorNetworkBase, self).__init__()
        self.d = d
        self.k = k

        self.proj1 = nn.Parameter(torch.randn(d, k)) # Oo
        self.proj2 = nn.Parameter(torch.randn(d, k)) # Rr
        self.proj3 = nn.Parameter(torch.randn(d, k)) # Ss

        self.core_tensor = nn.Parameter(torch.randn(k, k, k)) # ors

    def three_vectors_to_scalar(self, input1, input2, input3):
        # Step 1: Project inputs to k dimensions
        p1 = torch.einsum('Oo,bO->bo', self.proj1, input2) # bo
        p2 = torch.einsum('Rr,bR->br', self.proj2, input2) # br
        p3 = torch.einsum('Ss,bS->bs', self.proj3, input2) # bs

        # Step 2: Contract with the core tensor
        # Batch-wise contraction for a scalar output
        output = torch.einsum('bo,br,bs,ors->b', p1, p2, p3, self.core_tensor)
        return output

    def vector_to_matrix(self, input2):
        # Step 1: Project inputs to k dimensions
        p2 = torch.einsum('Rr,bR->br', self.proj2, input2) # br

        output = torch.einsum('Oo,ors,Ss,br->bos', self.proj1, self.core_tensor, self.proj3, p2)
        return output


class TensorNetworkVectorToMatrix(TensorNetworkBase):
    forward = TensorNetworkBase.vector_to_matrix


class TensorNetworkThreeVectorsToScalar(TensorNetworkBase):
    forward = TensorNetworkBase.three_vectors_to_scalar


# Example usage
d = 100  # Original input dimension
k = d // 10  # Reduced dimension

batch_size = 32
input1 = torch.randn(batch_size, d)
input2 = torch.randn(batch_size, d)
input3 = torch.randn(batch_size, d)

model = TensorNetworkVectorToMatrix(d, k)
output = model(input2)
print(output.shape)


model = TensorNetworkThreeVectorsToScalar(d, k)

output = model(input1, input2, input3)
print(output.shape)
