import os
import argparse
from glob import glob

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import Wav2Vec2ForCTC
from transformers import Wav2Vec2Processor
from utils.dataset import DataColletorTest
from transformers import Wav2Vec2FeatureExtractor

from utils.generic_utils import load_config, calculate_wer


def test(
    model,
    test_dataset,
    processor,
    calcule_wer,
    return_predictions,
    USE_CUDA,
    dataset_base_path
):
    model.eval()
    predictions = []
    tot_samples = 0
    tot_wer = 0
    tot_cer = 0
    with torch.no_grad():
        for batch in tqdm(test_dataset):
            input_values, attention_mask = batch['input_values'], batch['attention_mask']
            if calcule_wer:
                labels = batch['labels']

            if USE_CUDA:
                input_values = input_values.cuda(non_blocking=True)
                attention_mask = attention_mask.cuda(non_blocking=True)
                if calcule_wer:
                    labels = labels.cuda(non_blocking=True)

            logits = model(input_values, attention_mask=attention_mask).logits

            pred_ids = np.argmax(logits.cpu().detach().numpy(), axis=-1)

            if calcule_wer:
                # compute metrics
                wer, cer = calculate_wer(pred_ids, labels.cpu().detach().numpy(), processor, vocab_string)
                tot_wer += wer
                tot_cer += cer

            if return_predictions:
                audios_path = batch['audio_path']
                # get text
                pred_string = processor.batch_decode(pred_ids)

                for i in range(len(audios_path)):
                    output_wav_path = audios_path[i]
                    if dataset_base_path:
                        output_wav_path = output_wav_path.replace(dataset_base_path, '').replace(dataset_base_path+'/', '')

                    predictions.append([output_wav_path, pred_string[i].lower()])
            tot_samples += input_values.size(0)
    if calcule_wer:
        # calculate avg of metrics
        avg_wer = tot_wer/tot_samples
        avg_cer = tot_cer/tot_samples
        print("\n\n --> TEST PERFORMANCE\n")
        print("     | > :   WER    ({:.5f})\n".format(avg_wer))
        print("     | > :   CER    ({:.5f})\n".format(avg_cer))

    return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--checkpoint_path_or_name',
        type=str, required=True,
        help="path or name of checkpoints"
    )
    parser.add_argument(
        '-c',
        '--config_path',
        type=str,
        required=True,
        default=None,
        help="json file with configurations"
    )
    parser.add_argument(
        '--audio_path',
        type=str, default=None,
        help="If it's passed the inference will be done in all audio files in this path and the dataset present in the config json will be ignored")
    parser.add_argument(
        '--output_csv',
        type=str,
        default=None,
        help="CSV for save all predictions"
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help="Batch size"
    )

    args = parser.parse_args()

    config = load_config(args.config_path)

    # Use CUDA
    USE_CUDA = torch.cuda.is_available()

    model = Wav2Vec2ForCTC.from_pretrained(args.checkpoint_path_or_name)
    print("> Model Loaded")

    if config['apply_dbfs_norm']:
        print("> Using Gain Normalization")

    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=config['sampling_rate'],
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True
    )
    processor = Wav2Vec2Processor.from_pretrained(args.checkpoint_path_or_name)

    vocab_dict = processor.tokenizer.get_vocab()
     # if the model uses upper words in vocab force tokenizer lower case for compatibility with our data loader
    for c in list(vocab_dict.keys()):
        if c.isupper():
            processor.tokenizer.do_lower_case = True
            print("> Force lowercase Tokenizer !")
            break

    if USE_CUDA:
        model = model.cuda()

    dataset_base_path = None
    # use the datacolletor with only audio
    data_collator = DataColletorTest(
        processor=processor,
        sampling_rate=config.sampling_rate,
        padding=True,
        apply_dbfs_norm=config.apply_dbfs_norm,
        target_dbfs=config.target_dbfs
    )

    # load dataset
    print("\n\n> Searching Audios \n\n")
    wavs = glob(os.path.join(args.audio_path,'**/*.wav'), recursive=True)

    test_dataset = DataLoader(
        dataset=wavs,
        batch_size=args.batch_size,
        collate_fn=data_collator,
        shuffle=True,
        num_workers=config['num_loader_workers']
    )

    print("\n\n> Starting Evaluation \n\n")
    preds = test(
        model,
        test_dataset,
        processor,
        calcule_wer=False,
        return_predictions=True,
        USE_CUDA=USE_CUDA,
        dataset_base_path=dataset_base_path
    )

    df = pd.DataFrame(
        preds,
        columns=["file_path", "transcription"]
    )

    if args.output_csv:
        root_path = os.path.dirname(args.output_csv)
        os.makedirs(root_path, exist_ok=True)
        df.to_csv(args.output_csv, index=False)
        print("\n\n> Evaluation outputs saved in: ", args.output_csv)
    else:
        print(df.to_string())


if __name__ == '__main__':
    main()