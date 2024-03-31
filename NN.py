import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pymatgen import Structure, Composition

# Custom dataset class
class MaterialsDataset(Dataset):
    def __init__(self, data_file):
        self.data = pd.read_csv(data_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        formula = row["formula"]
        space_group = row["space_group"]
        stability = row["stability"]

        # Convert formula to composition object
        composition = Composition(formula)

        # Convert space group to one-hot encoding
        space_group_one_hot = torch.zeros(230)  # Assuming 230 space groups
        space_group_one_hot[space_group - 1] = 1

        # Convert stability to a tensor
        stability_tensor = torch.tensor([stability], dtype=torch.float32)

        return composition, space_group_one_hot, stability_tensor

# Model architecture
class MaterialsModel(nn.Module):
    def __init__(self):
        super(MaterialsModel, self).__init__()
        self.composition_embedding = nn.Embedding(num_embeddings=100, embedding_dim=32)
        self.space_group_embedding = nn.Linear(230, 16)
        self.fc1 = nn.Linear(32 + 16, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, composition, space_group):
        composition_embed = self.composition_embedding(composition)
        space_group_embed = self.space_group_embedding(space_group)
        x = torch.cat((composition_embed, space_group_embed), dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Training setup
dataset = MaterialsDataset("preprocessed_data.csv")
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
model = MaterialsModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    for composition, space_group, stability in dataloader:
        optimizer.zero_grad()
        outputs = model(composition, space_group)
        loss = criterion(outputs, stability)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

# Save the trained model
torch.save(model.state_dict(), "materials_model.pth")
