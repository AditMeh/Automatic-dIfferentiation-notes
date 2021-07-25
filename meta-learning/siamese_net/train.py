import torch
from torch.nn import BCELoss
from torch.optim import Adam
from torch.optim.sgd import SGD


def train(n_epochs, net, train_loader, learning_rate, device, batch_size):
    loss_fn = BCELoss(reduction='sum')
    optimizer = SGD(params=net.parameters(), lr=learning_rate)

    for epoch in range(1, n_epochs + 1):
        loss_train = 0.0

        for x1, x2, labels in train_loader:
            x1, x2, labels = x1.to(device=device), x2.to(
                device=device), labels.to(device=device)
            outputs = net(x1, x2)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train += loss.item()

        print('Epoch {}, Train Loss {}'.format(
            {epoch, loss_train/(batch_size * len(train_loader))}))
