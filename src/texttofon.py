# https://huggingface.co/hervezossou/speecht5_tts_fonvox

### Facebook

import torch
from transformers import VitsModel, AutoTokenizer
import scipy.io.wavfile
import os

class TextToFon:
    def __init__(self, model_name="facebook/mms-tts-fon"):
        """
        Initialize the Text-to-Fon class with model and tokenizer.
        """
        self.model = VitsModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.sampling_rate = self.model.config.sampling_rate

    def text_to_waveform(self, text):
        """
        Convert input text to waveform.
        """
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            output = self.model(**inputs).waveform
        return output

    def save_waveform(self, waveform, file_name="output.wav"):
        """
        Save the waveform to a wav file.
        """
        if not file_name.endswith('.wav'):
            file_name += '.wav'
        scipy.io.wavfile.write(file_name, rate=self.sampling_rate, data=waveform.squeeze().cpu().numpy())
        print(f"Waveform saved as {file_name}")

    def text_to_speech(self, text, output_file):
        """
        Converts text to speech and saves it to a specified wav file.
        """
        waveform = self.text_to_waveform(text)
        self.save_waveform(waveform, output_file)

# # Example usage
# if __name__ == "__main__":
#     tts = TextToFon()
#     text = "some example text in the Fon language"
#     tts.text_to_speech(text, "techno.wav")

# # use microsoft T5 pipeline

# from transformers import pipeline
# from datasets import load_dataset
# import soundfile as sf

# synthesiser = pipeline("text-to-speech", "hervezossou/speecht5_tts_fonvox")

# embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
# speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
# # You can replace this embedding with your own as well.

# speech = synthesiser("Hello, my dog is cooler than you!", forward_params={"speaker_embeddings": speaker_embedding})

# sf.write("speech.wav", speech["audio"], samplerate=speech["sampling_rate"])
