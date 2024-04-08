import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class MaterialsGNN(nn.Module):
    def __init__(self, num_elements, num_space_groups, hidden_dim=64, output_dim=1):
        super(MaterialsGNN, self).__init__()
        self.element_embed = nn.Embedding(num_elements, hidden_dim)
        self.space_group_embed = nn.Embedding(num_space_groups, hidden_dim)
        self.gnn_conv1 = GCNConv(hidden_dim, hidden_dim)
        self.gnn_conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, batch, space_group):
        # Element embedding
        x = self.element_embed(x)

        # Graph Convolutions
        x = self.relu(self.gnn_conv1(x, edge_index))
        x = self.relu(self.gnn_conv2(x, edge_index))

        # Global pooling
        x = global_mean_pool(x, batch)

        # Space group embedding
        space_group_embed = self.space_group_embed(space_group)

        # Concatenate features
        x = torch.cat([x, space_group_embed], dim=1)

        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class StabilityDiscriminator(nn.Module):
    def __init__(self, input_dim):
        super(StabilityDiscriminator, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim // 2)
        self.fc2 = nn.Linear(input_dim // 2, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

# Training setup
gnn = MaterialsGNN(num_elements=100, num_space_groups=230)
discriminator = StabilityDiscriminator(hidden_dim * 2)
criterion = nn.MSELoss()
g_optimizer = torch.optim.Adam(gnn.parameters(), lr=0.001)
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    # Train the generator (GNN)
    g_optimizer.zero_grad()
    x, edge_index, batch, space_group, y = # Load batch data
    generated_stability = gnn(x, edge_index, batch, space_group)
    g_loss = criterion(generated_stability, y)
    g_loss.backward()
    g_optimizer.step()

    # Train the discriminator
    d_optimizer.zero_grad()
    real_stability = # Get real stability data
    real_output = discriminator(real_stability)
    fake_output = discriminator(generated_stability.detach())
    d_loss = -torch.mean(torch.log(real_output + 1e-8)) - torch.mean(torch.log(1 - fake_output + 1e-8))
    d_loss.backward()
    d_optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Generator Loss: {g_loss.item():.4f}, Discriminator Loss: {d_loss.item():.4f}")

# Save the trained models
torch.save(gnn.state_dict(), "materials_gnn.pth")
torch.save(discriminator.state_dict(), "stability_discriminator.pth")
