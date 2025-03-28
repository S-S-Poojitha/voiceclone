from TTS.api import TTS
import numpy as np
import soundfile as sf

# Load XTTS v2 model
model = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

# Ensure reference audio is in WAV format
reference_audio_path = "converted_audio.wav"

# Define the text
text = """नमस्ते! मैं एक कृत्रिम बुद्धिमत्ता (AI) मॉडल हूँ, जिसे आपके लिए टेक्स्ट से स्पीच में बदलने के लिए डिज़ाइन किया गया है। 
मैं कई भाषाओं में बोल सकता हूँ और आपकी आवश्यकताओं के अनुसार ध्वनि उत्पन्न कर सकता हूँ। 
मशीन लर्निंग और डीप लर्निंग तकनीकों का उपयोग करके, मैं आपकी आवाज़ को समझ सकता हूँ और उसी अंदाज़ में बोलने की कोशिश कर सकता हूँ। 
आशा है कि यह वॉइस क्लोनिंग मॉडल आपके लिए उपयोगी साबित होगा। धन्यवाद!"""

# Generate speech using the reference speaker's voice
output_wav = model.tts(
    text=text,  
    speaker_wav=reference_audio_path,
    language="hi"
)

# Ensure output_wav is a NumPy array
output_wav = np.array(output_wav)  # Convert to NumPy array if needed

# Reshape if it's 1D (Mono audio)
if output_wav.ndim == 1:
    output_wav = output_wav.reshape(-1, 1)

# Save the output as a WAV file
sf.write("output.wav", output_wav, 24000)

print("Generated speech saved as output.wav")
