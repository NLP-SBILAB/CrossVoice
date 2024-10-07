# CrossVoice: Crosslingual Prosody Preserving Cascade-S2ST using Transfer Learning


This repository contains the code for our work titled `CrossVoice: Crosslingual Prosody Preserving Cascade-S2ST using Transfer Learning`, accepted at the Tiny Track of ICLR'24. The following scripts contain the code for the CrossVoice Pipeline for various hardware settings:

-  `crossvoice_fastwhisper_cpu.py`: Implementation of FastWhisper is used in CrossVoice which allows the pipeline to be run on the CPU as well.
-  `crossvoice_whisper_gpu.py`: Implementation of Whisper/Medium is used in CrossVoice which needs to be run on a GPU. Better performance is observed here.

If found helpful, please cite our work using
```
@inproceedings{
hira2024crossvoice,
title={CrossVoice: Crosslingual Prosody Preserving Cascade-S2{ST} using Transfer Learning},
author={Medha Hira and Arnav Goel and Anubha Gupta},
booktitle={The Second Tiny Papers Track at ICLR 2024},
year={2024},
url={https://openreview.net/forum?id=zEdBzTxXHl}
}
```
## Proposed Pipeline
<img width="885" alt="Screenshot 2024-10-07 at 1 01 23 PM" src="https://github.com/user-attachments/assets/028fe440-e9cf-4583-926b-3469a076e682">

## MoS Results
<img width="1397" alt="Screenshot 2024-10-07 at 1 41 48 PM" src="https://github.com/user-attachments/assets/ceb3f32b-9c45-4ec0-816b-98101b8877e7">
