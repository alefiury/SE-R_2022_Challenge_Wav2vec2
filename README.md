# SE&R 2022 Challenge - ASR Track

- [Introduction](#Introduction)
- [Dependencies](#Dependencies)
- [Inference](#Inference)
- [Pre-Trained Models](#Pre-Trained-Models)
- [Contact](#Contact)

# Introduction

This repository presents the code used for the submition on the SE&R 2022 Challenge (**GPED-CEIA-UFG** Team) and the paper [Domain Specific Wav2vec 2.0 Fine-tuning For The SE&R 2022 Challenge](http://ceur-ws.org/Vol-3175/paper02.pdf), based on the Edresson's repository [Wav2Vec-Wrapper](https://github.com/Edresson/Wav2Vec-Wrapper).

# Dependencies

It is important to install the dependencies before launching the application.

Run the following command to install the required dependencies using pip:

```
sudo pip install -r requeriments
```

# Inference

The inference can be made with a language model (using the ```predict_KenLM.py``` script) and without a language model (using the ```test.py``` script).

# Pre-Trained Models

The model weights with the best results can be found on the [huggingface hub](https://huggingface.co/alefiury/wav2vec2-large-xlsr-53-coraa-brazilian-portuguese-plus-gain-normalization).

# Author

- Alef Iury Siqueira Ferreira

# Citation

If you use this code for your research, please consider citing:

```
@inproceedings{Alef2022,
 title = {Domain Specific Wav2vec 2.0 Fine-tuning for the SE\&R 2022 Challenge},
 author = {Alef Iury Siqueira Ferreira and Gustavo dos Reis Oliveira},
 pages = {09--14},
 url = {http://ceur-ws.org/Vol-3175/paper02.pdf},
 crossref = {SER},
}
```

# Contact

- e-mail: alef_iury_c.c@discente.ufg.br
