import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class Model(nn.Module):
    def __init__(self, vocab_size, hidden_dim, output_dim, embedding_dim=128, dropout=0.3):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # BiLSTM doubles the hidden_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Must function for train model"""
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)  # Concatenate both directions
        output = self.fc(self.dropout(hidden))
        return output

    def train_model(self, train_dataloader, epochs=5, lr=0.001):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        for epoch in range(epochs):
            self.train()
            total_loss = 0
            for texts, labels in train_dataloader:
                optimizer.zero_grad()
                outputs = self(texts)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_dataloader)}")

    def evaluate(self, test_dataloader):
        """Evaluates the model and returns accuracy, precision, recall, and F1-score."""
        self.eval()
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for texts, labels in test_dataloader:
                outputs = self(texts)
                _, predicted = torch.max(outputs, 1)  # Get predicted class
                all_labels.extend(labels.tolist())  # Convert tensors to lists
                all_preds.extend(predicted.tolist())

        # Calculate evaluation metrics
        accuracy = accuracy_score(all_labels, all_preds) * 100
        precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
        recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
        f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1-score: {f1:.2f}")

        return accuracy, precision, recall, f1
