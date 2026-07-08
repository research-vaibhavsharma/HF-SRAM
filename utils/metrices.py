import torch

def evaluate_global_model(model, test_loader, device):
    """Evaluates the model on the global test set to log terminal accuracy."""
    model.eval()
    correct = 0
    total = 0
    loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss += criterion(outputs, y).item()
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            
    return (correct / total) * 100.0, loss / len(test_loader)
