import torch
import torch.nn.functional as F
from tqdm import tqdm


def train_epoch(model, optimizer, data_loader, device):
    model.train()
    total_loss = 0
    correct, count = 0, 0
    for batch in data_loader:
        optimizer.zero_grad()
        input_ids = batch["hidden"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["target"].to(device)

        logits = model(input_ids, attention_mask=attention_mask)
        loss = F.cross_entropy(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (labels == logits.argmax(dim=1)).int().sum().item()
        count += len(labels)

    avg_loss = total_loss / len(data_loader)
    accuracy = correct / count

    return avg_loss, accuracy


def valid_epoch(model, data_loader, device):
    with torch.no_grad():
        model.eval()
        valid_loss = 0
        correct, count = 0, 0
        for batch in data_loader:
            input_ids = batch["hidden"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["target"].to(device)

            logits = model(input_ids, attention_mask=attention_mask)
            loss = F.cross_entropy(logits, labels)

            valid_loss += loss.item()
            correct += (labels == logits.argmax(dim=1)).int().sum().item()
            count += len(labels)

        avg_loss = valid_loss / len(data_loader)
        accuracy = correct / count

    return avg_loss, accuracy


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def train_model(
    model, optimizer, data_loaders, num_epochs, device, save_path="best_model.pth"
):
    loss_history = {"Train": [], "Valid": []}
    accuracy_history = {"Train": [], "Valid": []}

    model.to(device)

    best_loss = 100

    pbar = tqdm(total=num_epochs, position=0, leave=True)
    early_stopper = EarlyStopper(patience=10, min_delta=0.005)

    for epoch in range(num_epochs):
        train_loss, train_accuracy = train_epoch(
            model, optimizer, data_loaders["Train"], device
        )
        valid_loss, valid_accuracy = valid_epoch(model, data_loaders["Valid"], device)

        pbar.set_description(
            f"Epoch {epoch + 1}/{num_epochs} - "
            f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
            f"Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_accuracy:.4f}"
        )

        loss_history["Train"].append(train_loss)
        accuracy_history["Train"].append(train_accuracy)

        loss_history["Valid"].append(valid_loss)
        accuracy_history["Valid"].append(valid_accuracy)

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), save_path)

        pbar.update(1)  # Update tqdm progress bar
        if early_stopper.early_stop(valid_loss):
            break

    model.load_state_dict(torch.load(save_path))

    return model, loss_history, accuracy_history
