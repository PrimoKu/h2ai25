--- 
license: apache-2.0
base_model: ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition
tags:
- generated_from_trainer
metrics:
- accuracy
model-index:
- name: xls-r-amharic
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

[<img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28.svg" alt="Visualize in Weights & Biases" width="200" height="32"/>](https://wandb.ai/hallo23/huggingface/runs/5pgjd6az)
# xls-r-amharic

This model is a fine-tuned version of [ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition](https://huggingface.co/ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 0.0901
- Accuracy: 0.9818

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 1e-05
- train_batch_size: 4
- eval_batch_size: 4
- seed: 42
- gradient_accumulation_steps: 2
- total_train_batch_size: 8
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 15
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch   | Step | Validation Loss | Accuracy |
|:-------------:|:-------:|:----:|:---------------:|:--------:|
| 0.2847        | 2.0202  | 500  | 0.2479          | 0.9212   |
| 0.1138        | 4.0404  | 1000 | 0.2063          | 0.9434   |
| 0.0614        | 6.0606  | 1500 | 0.1415          | 0.9657   |
| 0.0349        | 8.0808  | 2000 | 0.1383          | 0.9737   |
| 0.0143        | 10.1010 | 2500 | 0.0901          | 0.9818   |
| 0.0178        | 12.1212 | 3000 | 0.1188          | 0.9778   |
| 0.0222        | 14.1414 | 3500 | 0.1237          | 0.9778   |


### Framework versions

- Transformers 4.41.0.dev0
- Pytorch 2.1.2
- Datasets 2.19.1.dev0
- Tokenizers 0.19.1
