import torch
from torch.nn import BCELoss
from torch.optim import Adam
from torch.optim.sgd import SGD
from utils import plot_train_graph


def train(net, train_loader, val_loader, n_epochs, lr, device, batch_size):
    loss_fn = BCELoss(reduction='sum')
    optimizer = SGD(params=net.parameters(), lr=lr)
    val_history = []
    train_history = []

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

        loss_val = 0.0
        with torch.no_grad():
            for val_x1, val_x2, val_labels, in val_loader:
                val_x1, val_x2, val_labels = val_x1.to(device=device), val_x2.to(
                    device=device), val_labels.to(device=device)
                val_outputs = net(val_x1, val_x2)
                val_loss = loss_fn(val_outputs, val_labels)
 
                loss_val += val_loss

        train_loss_epoch = loss_train / (batch_size * len(train_loader))
        val_loss_epoch = loss_val/(batch_size * len(val_loader))

        train_history.append(train_loss_epoch)
        val_history.append(val_loss_epoch)

        print('Epoch {}, Train Loss {}, Val Loss {}'.format(
            epoch, train_loss_epoch, val_loss_epoch))

    plot_train_graph(Val_Loss=val_history, Train_Loss=train_history)
