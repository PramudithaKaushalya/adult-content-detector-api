import os
from fastapi import UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import torch
from pydantic import BaseModel
from torch.utils.data import DataLoader
from dataset import TextDataset, collate_fn, build_vocab, tokenize_text
from predict import predict
from model import Model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 10

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data.xlsx")
MODEL_PATH = os.path.join(BASE_DIR, "trained_model.pth")

PLOTS_DIR = "evaluation_plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

def train_and_save_model():
    train_data, test_data, vocab = read_data_set()

    # Create dataset and dataloader
    train_dataset = TextDataset(train_data, vocab)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)\

    # Initialize model with correct vocab size
    model = Model(vocab_size=len(vocab), hidden_dim=128, output_dim=2)

    # Train the model
    model.train_model(train_dataloader, epochs=EPOCHS)

    # Save the trained model
    torch.save(model.state_dict(), MODEL_PATH)
    print("Successfully saved the trained model to ", MODEL_PATH)
    return "Successfully trained and saved the model"

def load_trained_model(vocab):
    """Loads the trained model from disk and ensures compatibility."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model with correct vocab size
    model = Model(vocab_size=len(vocab), hidden_dim=128, output_dim=2)

    try:
        # Load model state dict
        state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()  # Set to evaluation mode
        print("Model successfully loaded from", MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)

    return model, device

def evaluate_model():
    train_data, test_data, vocab = read_data_set()
    print("Received request to evaluate model. 1")

    """Evaluates the trained model on the test dataset."""
    model, device = load_trained_model(vocab)
    print("Received request to evaluate model. 2")

    test_dataset = TextDataset(test_data, vocab)
    print("Received request to evaluate model. 3")
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    print("Received request to evaluate model. 4")

    # Evaluate the model and get all metrics
    accuracy, precision, recall, f1 = model.evaluate(test_dataloader)
    print("Received request to evaluate model. 5")

    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-score: {f1:.2f}")

    return f"Test Accuracy: {accuracy:.2f}%, \nPrecision: {precision:.2f}, \nRecall: {recall:.2f}, \nF1-score: {f1:.2f}"

def predict_from_model(input_text):
    train_data, test_data, vocab = read_data_set()

    """Runs predictions using the trained model."""
    model, device = load_trained_model(vocab)

    predicted_class, probabilities = predict(model, input_text, vocab)

    if predicted_class == 1:
        print(f"The text '{input_text}' is classified as **Adult Content** with probability {probabilities[0][1]:.2f}.")
        return {
            "predicted_class": predicted_class,
            "input_text": input_text,
            "description": f"This is classified as **Adult Content** with probability {probabilities[0][1]:.2f}."
        }
    else:
        print(f"The text '{input_text}' is classified as **Non-Adult Content** with probability {probabilities[0][0]:.2f}.")
        return {
            "predicted_class": predicted_class,
            "input_text": input_text,
            "description": f"This is classified as **Non-Adult Content** with probability {probabilities[0][0]:.2f}."
        }

def read_data_set():
    # Load dataset
    data = pd.read_excel(DATA_PATH)
    data = data[['text', 'label']]

    # Tokenize and prepare the dataset
    data = [(tokenize_text(row['text']), row['label']) for _, row in data.iterrows()]

    # Split dataset (70% train, 30% test)
    train_size = int(0.7 * len(data))
    train_data = data[:train_size]
    test_data = data[train_size:]

    # Build vocabulary
    vocab = build_vocab(train_data, min_freq=1)

    return train_data, test_data, vocab

def upload_and_append(file: UploadFile = File(...), sheet_name: str = "Data"):

    try:
        # Load uploaded file into a DataFrame
        if file.filename.endswith(".csv"):
            df_new = pd.read_csv(file.file)
        else:
            df_new = pd.read_excel(file.file, sheet_name=sheet_name, engine="openpyxl")

        # Clean column names
        df_new.columns = df_new.columns.str.strip().str.lower()

        # Load existing Excel file if it exists
        if os.path.exists(DATA_PATH):
            df_existing = pd.read_excel(DATA_PATH, engine="openpyxl")
            df_existing.columns = df_existing.columns.str.strip().str.lower()
            df_new = df_new[df_existing.columns]
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_combined = df_new

        # Save back to the same file
        df_combined.to_excel(DATA_PATH, index=False, engine="openpyxl")

        return f"{len(df_new)} records appended successfully."

    except ValueError as e:
        return {"error": f"Sheet '{sheet_name}' not found or unreadable."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class TextUploadRequest(BaseModel):
    text: str
    label: int

def append_record(record_obj : TextUploadRequest):
    # Convert class object to dict
    if not isinstance(record_obj, dict):
        try:
            record = vars(record_obj)
        except Exception:
            raise ValueError("Could not extract dictionary from object.")
    else:
        record = record_obj

    # Validate expected fields
    if "text" not in record or "label" not in record:
        raise ValueError("Record must contain 'text' and 'label' fields.")

    # Create a DataFrame from the new record
    new_df = pd.DataFrame([record])

    # Ensure column order and consistency
    if os.path.exists(DATA_PATH):
        existing_df = pd.read_excel(DATA_PATH, engine="openpyxl")
        existing_df.columns = existing_df.columns.str.strip().str.lower()
        new_df.columns = new_df.columns.str.strip().str.lower()

        # Reorder columns to match existing file
        new_df = new_df[existing_df.columns]
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = new_df

    # Save back to Excel
    combined_df.to_excel(DATA_PATH, index=False, engine="openpyxl")
    print("Record appended successfully.")

def fully_evaluate_model():
    train_data, test_data, vocab = read_data_set()
    model, device = load_trained_model(vocab)

    y_true, y_pred, y_probs = [], [], []

    for tokens, label in train_data:
        indices = [vocab.get(tok, vocab["<UNK>"]) for tok in tokens]
        tensor = torch.tensor(indices).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(tensor)
            probs = torch.softmax(output, dim=1).cpu().numpy()[0]
            pred = int(torch.argmax(output, dim=1).item())

        y_true.append(label)
        y_pred.append(pred)
        y_probs.append(probs)

    # Generate Metrics
    summary = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)

    # Confusion Matrix Plot
    fig_cm, ax = plt.subplots()
    ax.matshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], va='center', ha='center', color='black')

    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    cm_path = os.path.join(PLOTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close(fig_cm)

    # AUC-ROC Plot
    y_score = [prob[1] for prob in y_probs]  # probability of class 1
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    fig_roc, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    ax.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("AUC-ROC Curve")
    plt.legend()
    roc_path = os.path.join(PLOTS_DIR, "auc_roc_curve.png")
    plt.savefig(roc_path)
    plt.close(fig_roc)

    print(f"classification_report {summary}")
    return {
        "classification_report": summary
    }

def download_cm():
    return FileResponse(os.path.join(PLOTS_DIR, "confusion_matrix.png"), media_type="image/png")

def download_roc():
    return FileResponse(os.path.join(PLOTS_DIR, "auc_roc_curve.png"), media_type="image/png")