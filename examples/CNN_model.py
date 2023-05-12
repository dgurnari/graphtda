import numpy as np

from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm.auto import tqdm


class ConvNet(nn.Module):
    def __init__(self, dropout_prob=0, num_classes=2):
        super(ConvNet, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(2, 4, kernel_size=5, stride=2, padding=2), 
            nn.BatchNorm2d(4),
            nn.ReLU(True)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(8),
            nn.ReLU(True)
        )
        self.drop_out = nn.Dropout(p=dropout_prob)
        self.relu = torch.nn.ReLU()

        self.fc1 = nn.Linear(13 * 13 * 8, 100)
        self.fcOut = nn.Linear(100, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        # print(out.shape)
        out = self.layer2(out)
        # print(out.shape)

        out = out.reshape(out.size(0), -1)

        out = self.drop_out(out)

        out = self.fc1(out)
        out = self.drop_out(out)
        out = self.relu(out)
        out = self.fcOut(out)
        return out
    

# Use this function to evaluate your model
def accuracy(y_pred, y_true):
    """
    input y_pred: ndarray of shape (N,)
    input y_true: ndarray of shape (N,)
    """
    return (1.0 * (y_pred == y_true)).mean()


def train_CNN(X, y, num_folds=5, num_epochs=50, dropout_prob=0.5, lr = 1e-4, wd = 1e-5):
    train_accuracy_CNN = []
    test_accuracy_CNN = []

    skf = StratifiedKFold(n_splits=num_folds)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The model will be running on", device, "device")
            
    for train, test in tqdm(skf.split(X, y)):
        X_train = X[train]
        X_test = X[test]
        y_train = y[train]
        y_test = y[test]

        ### Training
        # initialize the net
        net = ConvNet(dropout_prob=dropout_prob, num_classes=len(np.unique(y)))
        # net = Network()

        # Move all the network parameters to the selected device
        net.to(device)

        ### Define the loss function
        loss_fn = nn.CrossEntropyLoss()

        ### Define an optimizer
        
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)

        # create the dataloaders
        train_data = torch.utils.data.TensorDataset(
            torch.tensor(X_train).float(), torch.tensor(y_train.astype(int).reshape(-1))
        )
        test_data = torch.utils.data.TensorDataset(
            torch.tensor(X_test).float(), torch.tensor(y_test.astype(int).reshape(-1))
        )

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=100, shuffle=False
        )

        for _ in range(num_epochs):
            # Training
            net.train()  # Training mode (e.g. enable dropout)
            # Eventually clear previous recorded gradients
            optimizer.zero_grad()

            for batch_idx, (train_data, train_label) in enumerate(train_loader):
                # Eventually clear previous recorded gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                #input_batch = train_data.unsqueeze(1).to(device)
                input_batch = train_data.to(device)

                outputs = net(input_batch)

                loss = loss_fn(outputs, train_label.to(device))

                loss.backward()
                optimizer.step()

        ## accuracy
        # train
        input_train = torch.tensor(X_train).float()
        label_train = y_train

        with torch.no_grad():
            # outputs = net(input_train.unsqueeze(1).to(device)).cpu()
            outputs = net(input_train.to(device)).cpu()
            softmax = nn.functional.softmax(outputs, dim = 1)
            predicted_label = np.argmax(softmax, axis=1).numpy()

        train_acc = accuracy(predicted_label, label_train)
        train_accuracy_CNN.append(train_acc)

        # test
        input_test = torch.tensor(X_test).float()
        label_test = y_test

        with torch.no_grad():
            # outputs = net(input_test.unsqueeze(1).to(device)).cpu()
            outputs = net(input_test.to(device)).cpu()
            softmax = nn.functional.softmax(outputs, dim = 1)
            predicted_label = np.argmax(softmax, axis=1).numpy()


        test_acc = accuracy(predicted_label, label_test)
        test_accuracy_CNN.append(test_acc)

        print('TRAIN {:.3}  TEST {:.3}'.format(train_acc, test_acc))

    return train_accuracy_CNN, test_accuracy_CNN