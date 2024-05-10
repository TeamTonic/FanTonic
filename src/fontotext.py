import torch
import soundfile as sf
from transformers import AutoModelForCTC, Wav2Vec2Processor

class FonToText:
    def __init__(self, model_name="OctaSpace/wav2vec2-bert-fongbe"):
        """
        Initializes the FonToText class.
        
        Parameters:
        - model_name (str): The Hugging Face model identifier.
        """
        # Automatically determine the device (GPU if available, otherwise CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the model and processor
        self.asr_model = AutoModelForCTC.from_pretrained(model_name).to(self.device)
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)

    def transcribe(self, audio_file_path):
        """
        Transcribes the given audio file to text.

        Parameters:
        - audio_file_path (str): The file path to the audio file (wav format).

        Returns:
        - str: The transcribed text.
        """
        # Read the audio file
        audio_input, sampling_rate = sf.read(audio_file_path)
        
        # Process the audio file
        inputs = self.processor(audio_input, sampling_rate=sampling_rate, return_tensors="pt")
        inputs = inputs.input_values.to(self.device)
        inputs = inputs.squeeze(1)
        
        with torch.no_grad():
            # Generate logits from the model
            logits = self.asr_model(inputs).logits
        
        # Find the predicted IDs and decode into text
        predicted_ids = torch.argmax(logits, dim=-1)
        predictions = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
        
        return predictions[0] if predictions else ""

# Example usage:
if __name__ == "__main__":
    audio_path = "path_to_your_audio_file.wav"
    transcriber = FonToText()
    print("Transcribed Text:", transcriber.transcribe(audio_path))