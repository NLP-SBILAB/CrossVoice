import argparse
from transformers import WhisperForConditionalGeneration, WhisperTokenizer
import torch
from datasets import load_metric
import soundfile as sf
import sacrebleu

def load_audio(audio_path):
    """ Load audio file. """
    audio, sr = sf.read(audio_path)
    return audio, sr

def transcribe_audio(audio, model, tokenizer, device):
    """ Transcribe audio using the Whisper model. """
    inputs = tokenizer(audio, return_tensors="pt", sampling_rate=16000).input_values.to(device)
    with torch.no_grad():
        logits = model.generate(inputs, max_length=512, temperature=1.0, do_sample=False)
    transcription = tokenizer.batch_decode(logits, skip_special_tokens=True)[0]
    return transcription

def calculate_bleu(reference, candidate, language):
    """ Calculate BLEU score between two transcriptions with language-specific adjustments. """
    if language in ["chinese", "japanese"]:
        # Tokenization for languages without spaces
        ref = [list(reference.replace(" ", ""))]
        cand = list(candidate.replace(" ", ""))
    else:
        # Tokenization for space-separated languages
        ref = [reference.split()]
        cand = candidate.split()

    bleu = sacrebleu.corpus_bleu([cand], [ref])
    return bleu.score

def parse_arguments():
    """ Parse command line arguments. """
    parser = argparse.ArgumentParser(description="Transcribe two audio clips and calculate BLEU scores.")
    parser.add_argument('audio_path1', type=str, help='Path to the first audio file.')
    parser.add_argument('audio_path2', type=str, help='Path to the second audio file.')
    parser.add_argument('language', type=str, help='Language of the audio clips.')
    return parser.parse_args()

def main():
    args = parse_arguments()

    # Set up model and tokenizer for Whisper
    model_name = f"openai/whisper-{args.language}-base"
    tokenizer = WhisperTokenizer.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load and transcribe the first audio file
    audio1, sr1 = load_audio(args.audio_path1)
    transcription1 = transcribe_audio(audio1, model, tokenizer, device)
    print(f"Transcription for Audio 1: {transcription1}")

    # Load and transcribe the second audio file
    audio2, sr2 = load_audio(args.audio_path2)
    transcription2 = transcribe_audio(audio2, model, tokenizer, device)
    print(f"Transcription for Audio 2: {transcription2}")

    # Calculate BLEU score between the transcriptions
    bleu_score = calculate_bleu(transcription1, transcription2)
    print(f"BLEU Score between the transcriptions: {bleu_score:.2f}")

if __name__ == '__main__':
    main()
