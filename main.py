import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class TensorNetworkBase(nn.Module):
    def __init__(self, d, k):
        super(TensorNetworkBase, self).__init__()
        self.d = d
        self.k = k

        self.proj1 = nn.Parameter(torch.randn(d, k)) # Ss
        self.proj2 = nn.Parameter(torch.randn(d, k)) # Rr
        self.proj3 = nn.Parameter(torch.randn(d, k)) # Oo

        self.core_tensor = nn.Parameter(torch.randn(k, k, k)) # sro

    def three_vectors_to_scalar(self, input1, input2, input3):
        # Step 1: Project inputs to k dimensions
        p1 = torch.einsum('Ss,bS->bs', self.proj1, input2) # bs
        p2 = torch.einsum('Rr,bR->br', self.proj2, input2) # br
        p3 = torch.einsum('Oo,bO->bo', self.proj3, input2) # bo

        # Step 2: Contract with the core tensor
        # Batch-wise contraction for a scalar output
        output = torch.einsum('bs,br,bo,sro->b', p1, p2, p3, self.core_tensor)
        return output

    # given input2, which is a relation, provide the matrix m
    # that maps input1, a subject, to output3, an object.
    # such a matrix has OS shape, and we have one for each element of the batch, hence bOS.
    def vector_to_matrix(self, input2):
        # Step 1: Project inputs to k dimensions
        p2 = torch.einsum('Rr,bR->br', self.proj2, input2) # br

        core_matrix = torch.einsum('ors,br->bso', self.core_tensor, p2) # bso
        matrix =  torch.einsum('Ss,bso,Oo->bOS', self.proj1, core_matrix, self.proj3)
        batch_size, d = input2.shape
        assert matrix.shape == (batch_size, d, d)
        return matrix


class TensorNetworkVectorToMatrix(TensorNetworkBase):
    forward = TensorNetworkBase.vector_to_matrix


class TensorNetworkThreeVectorsToScalar(TensorNetworkBase):
    forward = TensorNetworkBase.three_vectors_to_scalar



def test():
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




def train_tensor_network_vector_to_matrix(model, vectors, matrices, num_epochs=100, learning_rate=1e-3, batch_size=32):
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # Loss function (Mean Squared Error for regression, Frobenius norm)
    loss_fn = nn.MSELoss()

    N, d = vectors.shape

    for epoch in range(num_epochs):
        # Generate a random batch from the dataset
        batch_indices = torch.randint(0, N, (batch_size,))
        input_vectors = vectors[batch_indices]  # Shape: (batch_size, d)
        target_matrices = matrices[batch_indices]  # Shape: (batch_size, d, d)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        output_matrices = model(input_vectors)  # Shape: (batch_size, d, d)

        # Calculate loss
        loss = loss_fn(output_matrices, target_matrices)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Print loss for every epoch
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_training():
    d = 100
    k = 11

    # Instantiate the model for the vector-to-matrix prediction task
    model = TensorNetworkVectorToMatrix(d, k)

    print(f"Total number of parameters: {count_parameters(model)}")

    # Create dummy training data
    N = 47  # Number of samples
    vectors = torch.randn(N, d)  # Shape: (N, d)
    matrices = torch.randn(N, d, d)  # Shape: (N, d, d)

    # Train the model
    train_tensor_network_vector_to_matrix(model, vectors, matrices, num_epochs=100, learning_rate=1e-3, batch_size=32)


test_training()
