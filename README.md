# Duration Predictor and Learned Upsample - Parallel Tacotron 2
This is an unofficial implementation of the **Duration Predictor** and **Learned Upsample** modules in Parallel Tacotron 2 applying for FastSpeech 2. Specifically, these two modules are used to replace the **Variance Apdaptor** of FastSpeech 2.

## Preprocessing

LJSpeech dataset is used for training.
Please refer to [this repository](https://github.com/ming024/FastSpeech2) for more information about preprocessing.

## Training 

Training process can be ran by:
```
python train.py
```
## Remaining Issues
At the moment, model still cannot converge well.
Any comments or ieads to deal with the issue would be appreciated!

## References
* The backbone of this repository was adopted from [ming024's FastSpeech 2 - PyTorch Implementation](https://github.com/ming024/FastSpeech2).
* Dynamic time warping code is taken from [keonlee9420's Parallel-Tacotron2 implementation](https://github.com/keonlee9420/Parallel-Tacotron2)