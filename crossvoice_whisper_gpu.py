import torch
from TTS.api import TTS
from pydub import AudioSegment
from pydub.silence import split_on_silence
import noisereduce as nr
import librosa
import soundfile as sf
from googletrans import Translator
from faster_whisper import WhisperModel
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

"""
Dictionary to map languages to ISO codes
"""
lang_to_iso_tts = {
    "hindi": "hin",
    "marathi": "mar",
    "tamil": "tam",
    "telugu": "tel",
    "bengali": "ben",
    "gujarati": "guj",
    "kannada": "kan",
    "malayalam": "mal",
    "english": "eng",
    "french": "fra",
    "german": "deu",
    "italian": "ita",
    "dutch": "nld",
    "russian": "rus",
    "spanish": "spa",
    "indonesian": "ind",
    "chinese": "zho",
    "japanese": "jpn",
    "arabic": "ara"
}


lang_to_code_google_trans = {
    "hindi": "hi",
    "marathi": "mr",
    "tamil": "ta",
    "telugu": "te",
    "bengali": "bn",
    "gujarati": "gu",
    "kannada": "kn",
    "malayalam": "ml",
    "english": "en",
    "french": "fr",
    "german": "de",
    "italian": "it",
    "dutch": "nl",
    "russian": "ru",
    "spanish": "es",
    "indonesian": "id",
    "chinese": "zh",
    "japanese": "ja",
    "arabic": "ar"
}


"""
Function to translate text to a given language
"""
def translate_text(text, dest_language):
    translator = Translator()
    translation = translator.translate(text, dest=dest_language)
    return translation.text

"""
MAIN PIPELINE 
"""

if __name__ == "__main__":
    # Get device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # Get User Input for target language
    print("Enter the target language: ")
    langs = ["hindi", "marathi", "tamil", "telugu", "bengali", "gujarati", "kannada", "malayalam", "english"]
    for i, lang in enumerate(langs):
        print(f"{i+1}. {lang}")

    target_language_num = input("Enter the target language number: ")
    target_language = langs[int(target_language_num)-1]

    # get the ISO code for the target language
    # Check if the language is supported
    if target_language not in lang_to_iso_tts:
        print("Language not supported")
        exit(0)
    target_language_iso = lang_to_iso_tts[target_language]


    # get the google translate code for the target language
    target_language_google = lang_to_code_google_trans[target_language]

    """
    Load the Whisper Model here
    """

    model_id = "openai/whisper-large-v3"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, 
            torch_dtype=torch_dtype, 
            use_safetensors=True,
            attn_implementation="flash_attention_2"
        )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=512,
        chunk_length_s=35,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )

    """
    Enter Audio Path Here
    """
    audio_path = "/home/medhahira/Desktop/S2ST-Pipeline/S2ST-Pipeline/arnav_angry.wav"
    
    # # Input the audio path
    # audio_path = input("Enter the path to the audio file: ")

    """
    Audio Preprocessing - Silence Removal (Chunking)
    """

    audio_unproc = AudioSegment.from_file(audio_path)

    # Split audio on silences
    chunks = split_on_silence(
        audio_unproc,
        min_silence_len=50,  # Minimum length of silence to be considered (in ms)
        silence_thresh=-40   # Silence threshold in dB
    )

    # Combine chunks back into a single audio
    processed_audio = AudioSegment.silent(duration=0)
    for chunk in chunks:
        processed_audio += chunk

    # Export the result
    processed_audio.export("processed_audio.wav", format="wav")

    """
    Getting the Transcribed Text
    """
    result = pipe(audio_path, generate_kwargs={"temperature": 0.1, "do_sample": True})
    text = result["text"]

    """
    Running the Google Translate Model
    """
    print(text)
    translated_text = translate_text(text, target_language_google)
    print(translated_text)

    """
    Load the MMS TTS Model (need to put language ISO code)
    """

    api = TTS(f"tts_models/{target_language_iso}/fairseq/vits")

    api.tts_with_vc_to_file(
        text=translated_text,
        speaker_wav="processed_audio.wav",
        file_path= f"output_{target_language_google}.wav"
    )
