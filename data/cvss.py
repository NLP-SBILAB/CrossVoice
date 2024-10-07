import argparse
from datasets import load_dataset
import IPython.display as ipd

def load_cvss_dataset(language):
    """ Load the CVSS dataset for a specified language pair. """
    if language not in ['ar', 'zh', 'de', 'es', 'fr', 'it', 'ja', 'nl', 'ru', 'ta', 'hi']:
        raise ValueError(f"Unsupported language: {language}")
    cvss_c = load_dataset('google/cvss', 'cvss_c', languages=[language])
    return cvss_c

def load_common_voice_dataset(language):
    """ Load the Common Voice dataset for a specified language. """
    if language not in ['ar', 'zh', 'de', 'es', 'fr', 'it', 'ja', 'nl', 'ru', 'ta', 'hi']:
        raise ValueError(f"Unsupported language: {language}")
    common_voice = load_dataset('mozilla-foundation/common_voice_4_0', language)
    return common_voice

def display_audio(audio_path):
    """ Display an audio file. """
    print("Reference Audio: ")
    return ipd.Audio(audio_path)

def parse_arguments():
    """ Parse command line arguments. """
    parser = argparse.ArgumentParser(description="Load CVSS and Common Voice datasets for specified language.")
    parser.add_argument('--language', type=str, required=True, choices=['ar', 'zh', 'de', 'es', 'fr', 'it', 'ja', 'nl', 'ru', 'ta', 'hi'],
                        help='Language code for the dataset (e.g., "ar" for Arabic)')
    return parser.parse_args()

def main():
    args = parse_arguments()

    # Load CVSS dataset
    cvss_c = load_cvss_dataset(args.language)
    print("Loaded CVSS Dataset:")
    print(cvss_c)

    train_dataset = cvss_c['train']
    first_file = train_dataset[0]['id']
    first_audio = train_dataset[0]['audio']

    print(f"First file ID: {first_file}")
    print("First audio file info:")
    print(first_audio)
    audio_player = display_audio(first_audio['path'])

    # Load Common Voice dataset
    common_voice = load_common_voice_dataset(args.language)
    print("Loaded Common Voice Dataset:")
    print(common_voice)

if __name__ == '__main__':
    main()
