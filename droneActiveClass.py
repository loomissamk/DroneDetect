import os
import torch
import numpy as np
import torch.nn.functional as F
import librosa

# Device setup
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Load trained model
MODEL_PATH = "best_model.pt"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"[ERROR] Model file '{MODEL_PATH}' not found!")

model = torch.load(MODEL_PATH, map_location=device)
model.eval()

# Define confidence threshold
CONFIDENCE_THRESHOLD = 0.8

def ensure_iq_format(signal):
    """
    Converts the input signal into I/Q format if necessary.
    Expected shape: (2, N) where:
        - Row 0 = In-phase (I)
        - Row 1 = Quadrature (Q)
    
    If signal is not in I/Q format, attempt conversion.
    """
    signal = np.array(signal, dtype=np.float32)

    # Case 1: Already in (2, N) shape
    if signal.shape[0] == 2:
        return signal
    
    # Case 2: If mono signal (amplitude only), generate synthetic Q component
    elif len(signal.shape) == 1:
        print("[INFO] Converting amplitude-only signal to synthetic I/Q")
        return np.stack([signal, np.zeros_like(signal)])

    # Case 3: If stereo audio, assume Left = I, Right = Q
    elif signal.shape[0] == 2:
        print("[INFO] Converting stereo audio into I/Q format")
        return signal
    
    # Case 4: If a non-IQ format is detected, attempt transformation
    elif len(signal.shape) == 2 and signal.shape[1] == 2:
        print("[INFO] Converting 2-column signal into I/Q format")
        return signal.T  # Flip dimensions if needed

    else:
        raise ValueError(f"[ERROR] Unsupported signal format: {signal.shape}")


def predict_signal(signal):
    """
    Processes an incoming digitized signal, ensures it's in I/Q format, 
    passes it through the model, and applies confidence-based filtering.
    """
    try:
        # Ensure correct shape
        iq_signal = ensure_iq_format(signal)

        # Convert to Torch tensor and send to device
        iq_tensor = torch.tensor(iq_signal, dtype=torch.float32, device=device).unsqueeze(0)

        # Run through model
        with torch.no_grad():
            output = model(iq_tensor)
            probabilities = F.softmax(output, dim=1).cpu().numpy()[0]

        # Get the best prediction
        best_class = np.argmax(probabilities)
        best_confidence = probabilities[best_class]

        # Confidence filtering
        if best_confidence < CONFIDENCE_THRESHOLD:
            return "Unknown", best_confidence

        return best_class, best_confidence

    except Exception as e:
        print(f"[ERROR] Failed to process signal: {e}")
        return "Error", 0.0


# Example Testing
if __name__ == "__main__":
    test_signal = np.random.rand(2, 1048576)  # Example synthetic I/Q signal
    prediction, confidence = predict_signal(test_signal)
    print(f"[RESULT] Predicted Class: {prediction}, Confidence: {confidence:.2f}")
