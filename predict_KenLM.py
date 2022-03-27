import os
import glob
import argparse

import torch
import torchaudio
import numpy as np
import pandas as pd
from datasets import Dataset
from pydub import AudioSegment
from pyctcdecode import build_ctcdecoder
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

from utils.generic_utils import load_config


class STT:
    def __init__(
        self,
        model_name,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        lm=None
    ):

        self.model_name = model_name
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.vocab_dict = self.processor.tokenizer.get_vocab()
        self.sorted_dict = {k.lower(): v for k, v in sorted(self.vocab_dict.items(),key=lambda item: item[1])}
        self.device = device
        self.lm = lm
        if self.lm:
            self.lm_decoder = build_ctcdecoder(
                list(self.sorted_dict.keys()),
                self.lm
            )

    def batch_predict(self, batch):
        features = self.processor(batch["speech"],
                                  sampling_rate=batch["sampling_rate"][0],
                                  padding=True,
                                  return_tensors="pt")
        input_values = features.input_values.to(self.device)
        attention_mask = features.attention_mask.to(self.device)
        with torch.no_grad():
            logits = self.model(input_values, attention_mask=attention_mask).logits
        if self.lm:
            logits = logits.cpu().numpy()
            batch["predicted"] = []
            for sample_logits in logits:
                batch["predicted"].append(self.lm_decoder.decode(sample_logits))
        else:
            pred_ids = torch.argmax(logits, dim=-1)
            batch["predicted"] = self.processor.batch_decode(pred_ids)

        return batch


def map_to_array(batch, apply_norm, target_dbfs):
    if apply_norm:
         # Audio is loaded in a byte array
        sound = AudioSegment.from_file(batch["path"])
        change_in_dBFS = target_dbfs - sound.dBFS
        # Apply normalization
        normalized_sound = sound.apply_gain(change_in_dBFS)
        # Convert array of bytes back to array of samples in the range [-1, 1]
        # This enables to work wih the audio without saving on disk
        norm_audio_samples = np.array(normalized_sound.get_array_of_samples()).astype(np.float32, order='C') / 32768.0

        if sound.channels < 2:
            norm_audio_samples = np.expand_dims(norm_audio_samples, axis=0)

        # Expand one dimension and convert to torch tensor to have the save output shape and type as torchaudio.load
        speech = torch.from_numpy(norm_audio_samples)

    else:
        speech, _ = torchaudio.load(batch["path"])

    batch["speech"] = speech.squeeze(0).numpy()
    batch["sampling_rate"] = 16_000
    return batch


def load_data(dataset, num_workers, apply_norm, target_dbfs):
    return dataset.map(map_to_array,
                        num_proc=num_workers,
                        fn_kwargs={"apply_norm": apply_norm, "target_dbfs": target_dbfs})


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
        help="Path to where the audios are stored")
    parser.add_argument(
        '--output_csv',
        type=str,
        default=None,
        help="CSV to save the predictions"
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help="Batch size"
    )
    parser.add_argument(
        '--kenlm_path',
        type=str,
        default=None,
        help="Path to pretrained KenLM"
    )

    args = parser.parse_args()

    config = load_config(args.config_path)

    stt = STT(
        model_name=args.checkpoint_path_or_name,
        lm=args.kenlm_path
    )
    wav_paths = glob.glob(os.path.join(args.audio_path, '**', '*.wav'), recursive=True)

    df = pd.DataFrame(wav_paths, columns =['path'])
    ds = Dataset.from_pandas(df)

    loaded_ds = load_data(
        ds,
        num_workers=config['num_loader_workers'],
        apply_norm=config['apply_dbfs_norm'],
        target_dbfs=config['target_dbfs']
    )

    result = loaded_ds.map(
        stt.batch_predict,
        batched=True,
        batch_size=args.batch_size
    )

    paths = result['path']
    texts = result['predicted']

    df_pred = pd.DataFrame(list(zip(paths, texts)), columns =['file_path', 'transcription'])

    df_pred.to_csv(args.output_csv, index=False)
    print("\n\n> Evaluation outputs saved in: ", args.output_csv)


if __name__=='__main__':
    main()