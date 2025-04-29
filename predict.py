import torch
from dataset import tokenize_text

def predict(model, input_text, vocab):
    """
    Predict if a text is adult or non-adult based on the trained model.
    Parameters:
        model (Model): The trained model.
        input_text (str): The input text to classify.
        vocab (dict): The vocabulary used for tokenizing the input text.
    Returns:
        predicted_class (int): The predicted class (0 or 1).
        probabilities (tensor): The probabilities of each class.
    """
    # Tokenize the input text
    tokens = tokenize_text(input_text)

    # Convert tokens to indices, use '<UNK>' if word not in vocab
    token_indices = [vocab.get(token, vocab.get('<UNK>', 0)) for token in tokens]
    token_indices = torch.tensor(token_indices, dtype=torch.long).unsqueeze(0)  # Add batch dimension

    # Predict using the model
    model.eval()
    with torch.no_grad():
        outputs = model(token_indices)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

    return predicted_class, probabilities

