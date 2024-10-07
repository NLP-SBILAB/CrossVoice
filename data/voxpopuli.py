import argparse
from datasets import load_dataset

def load_voxpopuli_s2st_data(source_language, target_language, split='train'):
    """
    Load VoxPopuli dataset for speech-to-speech translation task.
    
    Args:
        source_language (str): Source language code ('fr' for French).
        target_language (str): Target language code ('en' for English).
        split (str): Dataset split to load ('train', 'validation', 'test').

    Returns:
        A tuple of datasets for the source and target languages.
    """
    # Validate input languages
    valid_languages = ['fr', 'en']
    if source_language not in valid_languages or target_language not in valid_languages:
        raise ValueError("Unsupported languages. Only 'fr' for French and 'en' for English are supported.")

    # Load the dataset for source and target languages
    source_data = load_dataset("voxpopuli", source_language, split=split)
    target_data = load_dataset("voxpopuli", target_language, split=split)
    
    return source_data, target_data

def parse_arguments():
    """ Parse command line arguments. """
    parser = argparse.ArgumentParser(description="Load VoxPopuli dataset for French to English S2ST tasks.")
    parser.add_argument('--split', type=str, default='train', choices=['train', 'validation', 'test'],
                        help='Dataset split to load (default: train)')
    return parser.parse_args()

def main():
    args = parse_arguments()

    # Load datasets for French to English S2ST
    fr_dataset, en_dataset = load_voxpopuli_s2st_data('fr', 'en', args.split)
    print(f"Loaded French {args.split} split with {len(fr_dataset)} samples.")
    print(f"Loaded English {args.split} split with {len(en_dataset)} samples.")

if __name__ == '__main__':
    main()
