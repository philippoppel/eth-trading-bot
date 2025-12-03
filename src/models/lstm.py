"""
LSTM Modell für Trend-Vorhersage.
Klassifiziert: Up (1), Down (-1), Sideways (0)
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from loguru import logger
from pathlib import Path


class LSTMModel(nn.Module):
    """LSTM Netzwerk für Zeitreihen-Klassifikation."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.2
    ):
        """
        Args:
            input_size: Anzahl Input Features
            hidden_size: LSTM Hidden Units
            num_layers: Anzahl LSTM Layers
            num_classes: Anzahl Output Klassen (3: up/down/side)
            dropout: Dropout Rate
        """
        super(LSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM Layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )

        # Attention Layer
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
            nn.Softmax(dim=1)
        )

        # Output Layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, x):
        """Forward Pass."""
        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq, hidden)

        # Attention
        attn_weights = self.attention(lstm_out)  # (batch, seq, 1)
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (batch, hidden)

        # Output
        out = self.fc(context)

        return out


def create_sequences(features: np.ndarray, labels: np.ndarray, seq_length: int = 60):
    """
    Erstellt Sequenzen für LSTM Training.

    Args:
        features: Feature Matrix (N, F)
        labels: Label Array (N,)
        seq_length: Länge der Sequenzen

    Returns:
        X: Sequenzen (N-seq, seq, F)
        y: Labels (N-seq,)
    """
    X, y = [], []

    for i in range(seq_length, len(features)):
        X.append(features[i-seq_length:i])
        y.append(labels[i])

    return np.array(X), np.array(y)


def create_labels(prices: np.ndarray, threshold: float = 0.005, lookahead: int = 4):
    """
    Erstellt Trend-Labels basierend auf zukünftiger Preisbewegung.

    Args:
        prices: Close Preise
        threshold: Mindestbewegung für Up/Down (0.5%)
        lookahead: Perioden in die Zukunft schauen

    Returns:
        Labels: 0=Down, 1=Sideways, 2=Up
    """
    labels = np.zeros(len(prices), dtype=int)

    for i in range(len(prices) - lookahead):
        future_return = (prices[i + lookahead] - prices[i]) / prices[i]

        if future_return > threshold:
            labels[i] = 2  # Up
        elif future_return < -threshold:
            labels[i] = 0  # Down
        else:
            labels[i] = 1  # Sideways

    # Letzte Einträge als Sideways markieren (keine Zukunft bekannt)
    labels[-lookahead:] = 1

    return labels


class LSTMTrainer:
    """Trainer für LSTM Modell."""

    def __init__(
        self,
        model: LSTMModel,
        learning_rate: float = 0.001,
        device: str = None
    ):
        """
        Args:
            model: LSTM Modell
            learning_rate: Lernrate
            device: 'cuda' oder 'cpu'
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        logger.info(f"LSTM Trainer initialisiert auf {self.device}")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        early_stopping_patience: int = 10,
        save_path: str = None
    ) -> dict:
        """
        Trainiert das Modell.

        Returns:
            History Dictionary mit Loss und Accuracy
        """
        # DataLoader erstellen
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.LongTensor(y_val)
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Training Loop
        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                train_correct += (predicted == y_batch).sum().item()
                train_total += y_batch.size(0)

            # Validation
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                    outputs = self.model(X_batch)
                    loss = self.criterion(outputs, y_batch)

                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    val_correct += (predicted == y_batch).sum().item()
                    val_total += y_batch.size(0)

            # Metriken berechnen
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            train_acc = train_correct / train_total
            val_acc = val_correct / val_total

            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)

            logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.4f} - "
                f"Val Loss: {avg_val_loss:.4f}, Acc: {val_acc:.4f}"
            )

            # Early Stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                if save_path:
                    self.save_model(save_path)
                    logger.info(f"Bestes Modell gespeichert: {save_path}")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early Stopping nach Epoch {epoch+1}")
                    break

        return history

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Evaluiert das Modell."""
        self.model.eval()

        X_tensor = torch.FloatTensor(X_test).to(self.device)
        y_tensor = torch.LongTensor(y_test).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)

            accuracy = (predicted == y_tensor).float().mean().item()

            # Class-wise Accuracy
            class_acc = {}
            for cls in range(3):
                mask = y_tensor == cls
                if mask.sum() > 0:
                    class_acc[cls] = (predicted[mask] == y_tensor[mask]).float().mean().item()

        return {
            'accuracy': accuracy,
            'class_accuracy': class_acc,
            'predictions': predicted.cpu().numpy()
        }

    def predict(self, X: np.ndarray) -> tuple:
        """
        Macht Vorhersagen.

        Returns:
            (predicted_class, probabilities)
        """
        self.model.eval()

        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

        return predicted.cpu().numpy(), probs.cpu().numpy()

    def save_model(self, path: str):
        """Speichert das Modell."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)

    def load_model(self, path: str):
        """Lädt das Modell."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Modell geladen: {path}")


if __name__ == "__main__":
    # Test
    input_size = 30
    seq_length = 60
    batch_size = 32

    # Dummy Daten
    X = np.random.randn(1000, seq_length, input_size).astype(np.float32)
    y = np.random.randint(0, 3, 1000)

    # Model
    model = LSTMModel(input_size=input_size)
    trainer = LSTMTrainer(model)

    # Split
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # Train
    history = trainer.train(X_train, y_train, X_val, y_val, epochs=5)

    # Evaluate
    result = trainer.evaluate(X_val, y_val)
    print(f"Accuracy: {result['accuracy']:.4f}")
