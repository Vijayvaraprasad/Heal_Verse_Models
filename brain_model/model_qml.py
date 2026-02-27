import torch
import torch.nn as nn
import pennylane as qml
import os

n_qubits = 4
n_layers = 2

qml_device_name = os.environ.get("QML_DEVICE", "default.qubit")
dev = qml.device(qml_device_name, wires=n_qubits)


# ================= FIXED QNODE =================
@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):

    # inputs shape: (n_qubits,)  ← single sample

    for layer in range(n_layers):

        for i in range(n_qubits):
            qml.RX(inputs[i], wires=i)

        for i in range(n_qubits):
            qml.RY(weights[layer][i][0], wires=i)
            qml.RZ(weights[layer][i][1], wires=i)

        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, (i + 1) % n_qubits])

    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


class QMLLayer(nn.Module):
    def __init__(self):
        super().__init__()
        weight_shapes = {"weights": (n_layers, n_qubits, 2)}
        self.qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

    def forward(self, x):

        # x shape: (batch_size, n_qubits)

        outputs = []
        for i in range(x.shape[0]):  # 🔥 manual batch handling
            outputs.append(self.qlayer(x[i]))

        return torch.stack(outputs)


class BrainCNN_QML(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fc1 = nn.Linear(128, n_qubits)
        self.qml = QMLLayer()
        self.fc2 = nn.Linear(n_qubits, 2)

    def forward(self, x):

        x = self.cnn(x)
        x = x.view(x.size(0), -1)

        x = self.fc1(x)

        x = self.qml(x)   # now safe

        x = self.fc2(x)

        return x