import argparse
import os
from pathlib import Path
import yaml
import torchaudio
from datasets import DatasetDict, Dataset, load_metric, Audio

_SAMPLE_RATE = 16000

def load_must_c_data(data_dir, language_pair):
    """
    Load and process the MuST-C dataset for the specified language pair.
    
    Args:
        data_dir (str): The root directory where MuST-C data is stored.
        language_pair (str): Language pair (e.g., 'en-de' or 'en-fr').

    Returns:
        A DatasetDict containing the splits as Dataset objects.
    """
    data_root = Path(data_dir) / language_pair / "data"
    splits = ["train", "dev", "tst-COMMON", "tst-HE"]
    dataset_dict = {}

    for split in splits:
        txt_dir = data_root / split / "txt"
        wav_dir = data_root / split / "wav"
        segments_path = txt_dir / f"{split}.yaml"

        with open(segments_path, 'r') as file:
            segments = yaml.safe_load(file)
        
        examples = []
        for segment in segments:
            audio_path = wav_dir / segment['wav']
            waveform, sr = torchaudio.load(audio_path, normalize=True)
            assert sr == _SAMPLE_RATE, "Sample rate mismatch"
            
            examples.append({
                'id': segment['id'],
                'audio': {'array': waveform.squeeze().numpy(), 'path': audio_path.as_posix(), 'sampling_rate': sr},
                'sentence': segment['sentence'],
                'translation': segment['translation']
            })

        dataset = Dataset.from_dict(examples)
        dataset_dict[split] = dataset

    return DatasetDict(dataset_dict)

def parse_arguments():
    """ Parse command line arguments. """
    parser = argparse.ArgumentParser(description="Load and process MuST-C dataset for speech translation.")
    parser.add_argument('--data_dir', type=str, required=True, help='Directory where MuST-C data is stored.')
    parser.add_argument('--language_pair', type=str, required=True, choices=['en-de', 'en-fr'],
                        help='Language pair to load (en-de for English to German, en-fr for English to French)')
    return parser.parse_args()

def main():
    args = parse_arguments()

    # Load the dataset
    dataset = load_must_c_data(args.data_dir, args.language_pair)
    print(f"Loaded data for {args.language_pair}:")
    for split, ds in dataset.items():
        print(f"{split}: {len(ds)} samples")

if __name__ == '__main__':
    main()
