from TTS.api import TTS

# Load XTTS v2 model
model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")

# Input text and reference audio
text = "नमस्ते, मैं एक AI हूँ।"  # Change to any Indian language
reference_audio_path = "22 Mar, 12.08 am_.m4a"  # Path to the speaker's voice

# Generate speech using the reference speaker's voice
output_wav = model.synthesize(
    text=text,
    speaker_wav=reference_audio_path,
    language="hi"  # Change to "ta" for Tamil, "te" for Telugu, etc.
)

# Save output
with open("output.wav", "wb") as f:
    f.write(output_wav)
print("Generated speech saved as output.wav")
