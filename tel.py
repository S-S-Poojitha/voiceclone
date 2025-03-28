from TTS.api import TTS
tts = TTS("tts_models/te/common_voice/vits")
tts.tts_to_file(text="నమస్తే! నేను ఒక AI మోడల్.", file_path="output_tel.wav")
