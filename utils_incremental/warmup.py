import torch
import torch.nn as nn
import torch.optim as optim


def warmup_model(model, dataloader, epochs=30, lr=0.01, device=None):
    if device is None:
        device = next(model.parameters()).device
    model.to(device)
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for inputs, targets, _, _ in dataloader:
            inputs = inputs.to(device)
            targets = targets.argmax(dim=1).to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    return model