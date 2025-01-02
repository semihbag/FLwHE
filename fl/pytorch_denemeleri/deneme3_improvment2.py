import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, random_split

# Modelin tanımlanması
class ImprovedModel(nn.Module):
    def _init_(self):
        super(ImprovedModel, self)._init_()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)  # 64 kanallı ve 8x8 boyutlu çıktı
        self.fc2 = nn.Linear(256, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Federated client sınıfı
class FederatedClient:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)

    def train(self, epochs):
        self.model.train()
        for epoch in range(epochs):
            for images, labels in self.data:
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = F.cross_entropy(outputs, labels)
                loss.backward()
                self.optimizer.step()

    def get_weights(self):
        return {name: param.data for name, param in self.model.named_parameters()}

    def set_weights(self, weights):
        for name, param in self.model.named_parameters():
            param.data = weights[name]

# Test için veri yükleme
def get_test_data_loader():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    test_data = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    return DataLoader(test_data, batch_size=32)

# Otomatik veri yükleme ve dağıtma
def get_data_loaders(num_clients):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    # Veriyi eşit şekilde ve homojen olarak her bir istemciye dağıtma
    data_per_client = len(train_data) // num_clients
    clients_data = random_split(train_data, [data_per_client] * num_clients)
    
    # DataLoader'lar oluşturma
    return [DataLoader(client_data, batch_size=32) for client_data in clients_data]

# Modelin doğruluğunu hesaplama
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    average_loss = total_loss / len(test_loader)
    return accuracy, average_loss

# Ana fonksiyon
def main():
    # Dinamik değişkenler
    num_clients = 5  # İstemci sayısı
    epochs = 1       # Epoch sayısı
    rounds = 5      # Round sayısı

    # Veri yükleme
    client_data_loaders = get_data_loaders(num_clients)
    test_loader = get_test_data_loader()

    # Model oluşturma
    model = ImprovedModel()
    
    # İstemcileri oluşturma
    clients = [FederatedClient(model, client_data) for client_data in client_data_loaders]

    # Federated learning süreci
    for round_num in range(rounds):
        print(f"Round {round_num + 1}/{rounds}")
        for client in clients:
            client.train(epochs=epochs)
        
        # Ağırlıkları al ve ortalama al
        averaged_weights = {}
        for client in clients:
            client_weights = client.get_weights()
            for key in client_weights.keys():
                if key not in averaged_weights:
                    averaged_weights[key] = client_weights[key]
                else:
                    averaged_weights[key] += client_weights[key]
        
        # Ağırlıkları ortalamak
        for key in averaged_weights.keys():
            averaged_weights[key] /= num_clients
        
        # Modelin ağırlıklarını güncelle
        model.load_state_dict(averaged_weights)

        # Modelin performansını değerlendir
        accuracy, average_loss = evaluate_model(model, test_loader)
        print(f"Accuracy: {accuracy:.2f}%, Loss: {average_loss:.4f}")

    print("Federated learning tamamlandı.")

if __name__ == "__main__":
    main()
