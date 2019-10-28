# End-to-end speech recognition on Pytorch

python3 generate_fastalign_alignment.py --en_path SEAME_PhaseII_dataset/lm_generation/google\ translate/train_clean.en --zh_path SEAME_PhaseII_dataset/lm_generation/google\ translate/train_clean.zh --output_path SEAME_PhaseII_dataset/lm_generation/fastalign_generated/eq_train_3best.txt --equal_constraint

python3 generate_fastalign_alignment.py --en_path SEAME_PhaseII_dataset/lm_generation/google\ translate/train_clean.en --zh_path SEAME_PhaseII_dataset/lm_generation/google\ translate/train_clean.zh --output_path SEAME_PhaseII_dataset/lm_generation/fastalign_generated/random_train_3best.txt







Implementation of Transformer ASR

Translate
```
python3 data/translate_google.py
python3 data/clean_translation_google.py

python3 clean_translation_google.py --input_path SEAME_PhaseII_dataset/lm_generation/google\ translate/train.en --output_path SEAME_PhaseII_dataset/lm_generation/google\ translate/train_clean.en

```




Train
```
CUDA_VISIBLE_DEVICES=2 python3 train.py --train-manifest data/manifests/seame_phaseII_train_manifest.csv --val-manifest data/manifests/seame_phaseII_val_manifest.csv --test-manifest data/manifests/seame_phaseII_test_manifest.csv --cuda --batch-size 8 --labels-path data/labels/hkust_cv_seame.json --lr 1e-4 --name train_seame_16khz_drop0.1_cnn_batch8_layers3_heads8_lr1e-4_shuffle --save-folder save/ --save-every 1 --emb_cnn --dropout 0.1 --num-layers 3 --num-heads 8 --dim-model 256 --dim-key 64 --dim-value 64 --dim-input 161 --dim-inner 512 --dim-emb 256 --epochs 2000 --src-max-len 2000 --tgt-max-len 1000 --shuffle
```

Multi-train
```
CUDA_VISIBLE_DEVICES=0,1 python3 multi_train.py --train-manifest-list data/manifests/cv-valid-train_manifest.csv data/manifests/hkust_16khz_train_manifest.csv --val-manifest-list data/manifests/cv-valid-dev_manifest.csv data/manifests/hkust_16khz_val_manifest.csv --test-manifest data/manifests/hkust_16khz_val_manifest.csv --cuda --batch-size 8 --labels-path data/labels/hkust_cv_seame.json --lr 1e-4 --name multi_train_16khz_drop0.1_cnn_batch8_layers3_heads8_lr1e-4_shuffle --save-folder save/ --save-every 1 --emb_cnn --dropout 0.1 --num-layers 3 --num-heads 8 --dim-model 256 --dim-key 64 --dim-value 64 --dim-input 161 --dim-inner 512 --dim-emb 256 --epochs 2000 --src-max-len 2000 --tgt-max-len 1000 --parallel --device-ids 0 1 --shuffle
```

Finetune
```
CUDA_VISIBLE_DEVICES=2 python3 train.py --train-manifest data/manifests/seame_phaseII_train_manifest.csv --val-manifest data
/manifests/seame_phaseII_val_manifest.csv --test-manifest data/manifests/seame_phaseII_test_manifest.csv --cuda --batch-size 8 --labels-path data/labels/hkust_cv_seame.json
 --lr 1e-4 --name finetune_train_seame_16khz_drop0.1_cnn_batch8_layers3_heads8_lr1e-4_shuffle --save-folder save/ --save-every 1 --emb_cnn --dropout 0.1 --num-layers 3 --nu
m-heads 8 --dim-model 256 --dim-key 64 --dim-value 64 --dim-input 161 --dim-inner 512 --dim-emb 256 --epochs 2000 --src-max-len 2000 --tgt-max-len 1000 --shuffle --continue
-from save/multi_train_16khz_drop0.1_cnn_batch8_layers3_heads8_lr1e-4_shuffle/epoch_18.th
```








CTC
```
 CUDA_VISIBLE_DEVICES=0 python3 train.py --train-manifest data/manifests/seame_phaseII_train_manifest.csv --val-manifest data/manifests/seame_phaseII_val_manifest.csv --test-manifest data/manifests/seame_phaseII_test_manifest.csv --cuda --batch-size 8 --labels-path data/labels/hkust_cv_seame.json --lr 1e-4 --name train_seame_16hkz_deepspeech_layers4_dim400 --save-folder save/ --save-every 1 --emb_cnn --dropout 0.1 --num-layers 4 --dim-model 400 --epochs 2000 --model=DEEPSPEECH --shuffle --loss=ctc --clip
```

 CUDA_VISIBLE_DEVICES=1 python3 train.py --train-manifest data/manifests/seame_phaseII_train_manifest.csv --val-manifest data/manifests/seame_phaseII_val_manifest.csv --test-manifest data/manifests/seame_phaseII_test_manifest.csv --cuda --batch-size 8 --labels-path data/labels/hkust_cv_seame.json --lr 1e-4 --name train_seame_16hkz_deepspeech_layers2_dim200 --save-folder save/ --save-every 1 --emb_cnn --dropout 0.1 --num-layers 2 --dim-model 200 --epochs 2000 --model=DEEPSPEECH --shuffle --loss=ctc --clip






?
CUDA_VISIBLE_DEVICES=3 python3 train_lm.py --cuda --verbose --batch-size 4 --train-manifest data/manifests/seame_phaseII_train_manifest.csv --val-manifest data/manifests/seame_phaseII_val_manifest.csv --labels-path data/labels/hkust_cv_seame.json --dim-model 512 --num-head 8 --dim-value 128














Train SEAME
```
CUDA_VISIBLE_DEVICES=0 python3 train.py --train-manifest data/manifests/seame_phaseII_8khz_train_manifest.csv --val-manifest data/manifests/seame_phaseII_8khz_val_manifest.csv --test-manifest data/manifests/seame_phaseII_8khz_test_manifest.csv --cuda --batch-size 8 --labels-path data/labels/hkust_16khz_aishell_cv_valid_labels.json --lr 1e-4 --name train_seame_8khz_drop0.1_cnn_batch8_layers3_heads8_lr1e-4 --save-folder save/ --save-every 1 --emb_cnn --dropout 0.1 --num-layers 3 --num-heads 8 --dim-model 256 --dim-key 64 --dim-value 64 --dim-input 161 --dim-inner 512 --dim-emb 256 --epochs 2000 --src-max-len 2000 --tgt-max-len 1000 --sample-rate 8000
```
Test
```
CUDA_VISIBLE_DEVICES=3 python3 test.py --test-manifest data/manifests/seame_phaseII_8khz_test_manifest.csv --continue-from save/train_seame_8khz_drop0.1_cnn_batch8_layers3_heads8_lr1e-4/epoch_7.th --verbose --cuda
```

```
CUDA_VISIBLE_DEVICES=1 python3 train.py --train-manifest data/manifests/seame_phaseII_8khz_train_manifest.csv --val-manifest data/manifests/seame_phaseII_8khz_val_manifest.csv --test-manifest data/manifests/seame_phaseII_8khz_test_manifest.csv --cuda --batch-size 8 --labels-path data/labels/hkust_16khz_aishell_cv_valid_labels.json --lr 1e-4 --name train_seame_8khz_drop0.2_cnn_batch8_layers3_heads8_lr1e-4 --save-folder save/ --save-every 1 --emb_cnn --dropout 0.2 --num-layers 3 --num-heads 8 --dim-model 256 --dim-key 64 --dim-value 64 --dim-input 161 --dim-inner 512 --dim-emb 256 --epochs 2000 --src-max-len 2000 --tgt-max-len 1000 --sample-rate 8000
```

```
CUDA_VISIBLE_DEVICES=4 python3 train.py --train-manifest data/manifests/seame_phaseII_8khz_train_manifest.csv --val-manifest data/manifests/seame_phaseII_8khz_val_manifest.csv --test-manifest data/manifests/seame_phaseII_8khz_test_manifest.csv --cuda --batch-size 12 --labels-path data/labels/hkust_16khz_aishell_cv_valid_labels.json --lr 1e-4 --name train_seame_8khz_drop0.1_cnn_batch12_layers3_heads8_lr1e-4 --save-folder save/ --save-every 1 --emb_cnn --dropout 0.1 --num-layers 3 --num-heads 8 --dim-model 256 --dim-key 64 --dim-value 64 --dim-input 161 --dim-inner 512 --dim-emb 256 --epochs 2000 --src-max-len 2000 --tgt-max-len 1000 --sample-rate 8000
```

```
CUDA_VISIBLE_DEVICES=3 python3 train.py --train-manifest data/manifests/seame_phaseII_8khz_train_manifest.csv --val-manifest data/manifests/seame_phaseII_8khz_val_manifest.csv --test-manifest data/manifests/seame_phaseII_8khz_test_manifest.csv --cuda --batch-size 8 --labels-path data/labels/hkust_16khz_aishell_cv_valid_labels.json --lr 1e-4 --name train_seame_8khz_drop0.1_cnn_batch8_layers3_heads4_lr1e-4_model512 --save-folder save/ --save-every 1 --emb_cnn --dropout 0.1 --num-layers 3 --num-heads 4 --dim-model 512 --dim-key 64 --dim-value 64 --dim-input 161 --dim-inner 512 --dim-emb 512 --epochs 2000 --src-max-len 2000 --tgt-max-len 1000 --sample-rate 8000
```

CUDA_VISIBLE_DEVICES=0 python3 train.py --train-manifest data/manifests/seame_phaseII_train_manifest.csv --val-manifest data/manifests/seame_phaseII_val_manifest.csv --test-manifest data/manifests/seame_phaseII_test_manifest.csv --cuda --batch-size 8 --labels-path data/labels/hkust_16khz_aishell_cv_valid_labels.json --lr 1e-4 --name train_seame_drop0.1_cnn_batch8_layers3_heads8_lr1e-4_shuffle --save-folder save/ --save-every 1 --emb_cnn --dropout 0.1 --num-layers 3 --num-heads 8 --dim-model 256 --dim-key 64 --dim-value 64 --dim-input 161 --dim-inner 512 --dim-emb 256 --epochs 2000 --src-max-len 2000 --tgt-max-len 1000 --shuffle

CUDA_VISIBLE_DEVICES=1 python3 train.py --train-manifest data/manifests/seame_phaseII_8khz_train_manifest.csv --val-manifest data/manifests/seame_phaseII_8khz_val_manifest.csv --test-manifest data/manifests/seame_phaseII_8khz_test_manifest.csv --cuda --batch-size 8 --labels-path data/labels/hkust_16khz_aishell_cv_valid_labels.json --lr 1e-4 --name train_seame_8khz_drop0.1_cnn_batch8_layers3_heads4_lr1e-4_model512_shuffle --save-folder save/ --save-every 1 --emb_cnn --dropout 0.1 --num-layers 3 --num-heads 4 --dim-model 512 --dim-key 64 --dim-value 64 --dim-input 161 --dim-inner 512 --dim-emb 512 --epochs 2000 --src-max-len 2000 --tgt-max-len 1000 --sample-rate 8000 --shuffle

Multi-train
```
CUDA_VISIBLE_DEVICES=2,3 python3 multi_train.py --train-manifest-list data/manifests/cv_valid_8khz_train_manifest.csv data/manifests/hkust_train_manifest.csv --val-manifest-list data/manifests/cv_valid_8khz_val_manifest.csv data/manifests/hkust_val_manifest.csv --test-manifest data/manifests/hkust_val_manifest.csv --cuda --batch-size 8 --labels-path data/labels/hkust_16khz_aishell_cv_valid_labels.json --lr 1e-4 --name multi_train_8khz_drop0.1_cnn_batch8_layers3_heads8_lr1e-4 --save-folder save/ --save-every 1 --emb_cnn --dropout 0.1 --num-layers 3 --num-heads 8 --dim-model 256 --dim-key 64 --dim-value 64 --dim-input 161 --dim-inner 512 --dim-emb 256 --epochs 2000 --src-max-len 2000 --tgt-max-len 1000  --sample-rate 8000 --parallel --device-ids 0 1
```

```
CUDA_VISIBLE_DEVICES=2,3 python3 multi_train.py --train-manifest-list data/manifests/cv-valid-train_manifest.csv data/manifests/hkust_16khz_train_manifest.csv --val-manifest-list data/manifests/cv-valid-dev_manifest.csv data/manifests/hkust_16khz_val_manifest.csv --test-manifest data/manifests/hkust_16khz_val_manifest.csv --cuda --batch-size 8 --labels-path data/labels/hkust_16khz_aishell_cv_valid_labels.json --lr 1e-4 --name multi_train_drop0.1_cnn_batch8_layers3_heads8_lr1e-4_shuffle --save-folder save/ --save-every 1 --emb_cnn --dropout 0.1 --num-layers 3 --num-heads 8 --dim-model 256 --dim-key 64 --dim-value 64 --dim-input 161 --dim-inner 512 --dim-emb 256 --epochs 2000 --src-max-len 2000 --tgt-max-len 1000 --parallel --device-ids 0 1 --shuffle
```



























## Train
Train with joint-training (batch consists of balanced samples)
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 multi_train.py --train-manifest-list data/manifests/cv-valid-train_manifest.csv data/manifests/hkust_16khz_train_manifest.csv --val-manifest-list data/manifests/cv-valid-dev_manifest.csv data/manifests/hkust_16khz_val_manifest.csv --test-manifest data/manifests/cv-valid-test_manifest.csv --cuda --batch-size 16 --labels-path data/labels/hkust_16khz_aishell_cv_valid_labels.json --lr 1e-4 --name multi_train_drop0.1_cnn_batch16_layers5_heads8 --save-folder save/ --save-every 1 --emb_cnn --dropout 0.1 --num-layers 5 --num-heads 8 --dim-model 512 --dim-key 64 --dim-value 64 --dim-input 161 --dim-inner 1024 --dim-emb 512 --epochs 2000 --parallel --device-ids 0 1 2 3 --src-max-len 2000 --tgt-max-len 1000
```

```
CUDA_VISIBLE_DEVICES=0,1 python3 multi_train.py --train-manifest-list data/manifests/cv-valid-train_manifest.csv data/manifests/hkust_16khz_train_manifest.csv --val-manifest-list data/manifests/cv-valid-dev_manifest.csv data/manifests/hkust_16khz_val_manifest.csv --test-manifest data/manifests/cv-valid-test_manifest.csv --cuda --batch-size 8 --labels-path data/labels/hkust_16khz_aishell_cv_valid_labels.json --lr 1e-4 --name multi_train_drop0.1_cnn_batch8_layers3_heads8_lr1e-4 --save-folder save/ --save-every 1 --emb_cnn --dropout 0.1 --num-layers 3 --num-heads 8 --dim-model 256 --dim-key 64 --dim-value 64 --dim-input 161 --dim-inner 512 --dim-emb 256 --epochs 2000 --src-max-len 2000 --tgt-max-len 1000 --parallel --device-ids 0 1
```
Dummy
```
CUDA_VISIBLE_DEVICES=4 python3 multi_train.py --train-manifest-list data/manifests/cv-valid-dev_manifest.csv data/manifests/hkust_16khz_val_manifest.csv --val-manifest-list data/manifests/cv-valid-dev_manifest.csv data/manifests/hkust_16khz_val_manifest.csv --test-manifest data/manifests/cv-valid-test_manifest.csv --cuda --batch-size 4 --labels-path data/labels/hkust_16khz_aishell_cv_valid_labels.json --lr 1e-4 --name multi_train_drop0.1_cnn_batch4_layers5_heads8 --save-folder save/ --save-every 1 --emb_cnn --dropout 0.1 --num-layers 5 --num-heads 8 --dim-model 512 --dim-key 64 --dim-value 64 --dim-input 161 --dim-inner 1024 --dim-emb 512 --epochs 2000 --src-max-len 2000 --tgt-max-len 1000
```

```
CUDA_VISIBLE_DEVICES=2 python3 multi_train.py --train-manifest-list data/manifests/cv-valid-dev_manifest.csv data/manifests/hkust_16khz_val_manifest.csv --val-manifest-list data/manifests/cv-valid-dev_manifest.csv data/manifests/hkust_16khz_val_manifest.csv --test-manifest data/manifests/cv-valid-test_manifest.csv --cuda --batch-size 4 --labels-path data/labels/hkust_16khz_aishell_cv_valid_labels.json --lr 1e-4 --name dummy_multi_train_drop0.1_cnn_batch8_layers3_heads8_lr1e-4 --save-folder save/ --save-every 1 --emb_cnn --dropout 0.1 --num-layers 3 --num-heads 8 --dim-model 256 --dim-key 64 --dim-value 64 --dim-input 161 --dim-inner 512 --dim-emb 256 --epochs 2000 --src-max-len 2000 --tgt-max-len 1000
```

Resume
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 multi_train.py --train-manifest-list data/manifests/cv-valid-train_manifest.csv data/manifests/hkust_16khz_train_manifest.csv --val-manifest-list data/manifests/cv-valid-dev_manifest.csv data/manifests/hkust_16khz_val_manifest.csv --test-manifest data/manifests/cv-valid-test_manifest.csv --cuda --batch-size 16 --labels-path data/labels/hkust_16khz_aishell_cv_valid_labels.json --lr 1e-4 --name multi_train_drop0.1_cnn_batch16_layers5_heads8 --save-folder save/ --save-every 1 --emb_cnn --dropout 0.1 --num-layers 5 --num-heads 8 --dim-model 512 --dim-key 64 --dim-value 64 --dim-input 161 --dim-inner 1024 --dim-emb 512 --epochs 2000 --parallel --device-ids 0 1 2 3 --src-max-len 2000 --tgt-max-len 1000 --continue-from save/multi_train_drop0.1_cnn_batch16_layers5_heads8/epoch_7.th
```

Train only with seame
```
CUDA_VISIBLE_DEVICES=3 python3 train.py --train-manifest data/manifests/seame_phaseII_train_manifest.csv --val-manifest data/manifests/seame_phaseII_val_manifest.csv --test-manifest data/manifests/seame_phaseII_test_manifest.csv --cuda --batch-size 8 --labels-path data/labels/hkust_16khz_aishell_cv_valid_labels.json --lr 1e-4 --name train_seame_drop0.1_cnn_batch8_layers5_heads8 --save-folder save/ --save-every 1 --emb_cnn --dropout 0.1 --num-layers 5 --num-heads 8 --dim-model 512 --dim-key 64 --dim-value 64 --dim-input 161 --dim-inner 1024 --dim-emb 512 --epochs 2000 --src-max-len 2000 --tgt-max-len 1000 (slow to converge)
```

```
CUDA_VISIBLE_DEVICES=3 python3 train.py --train-manifest data/manifests/seame_phaseII_train_manifest.csv --val-manifest data/manifests/seame_phaseII_val_manifest.csv --test-manifest data/manifests/seame_phaseII_test_manifest.csv --cuda --batch-size 8 --labels-path data/labels/hkust_16khz_aishell_cv_valid_labels.json --lr 1e-4 --name train_seame_drop0.1_cnn_batch8_layers3_heads8_lr1e-4 --save-folder save/ --save-every 1 --emb_cnn --dropout 0.1 --num-layers 3 --num-heads 8 --dim-model 256 --dim-key 64 --dim-value 64 --dim-input 161 --dim-inner 512 --dim-emb 256 --epochs 2000 --src-max-len 2000 --tgt-max-len 1000
```

```
CUDA_VISIBLE_DEVICES=4 python3 train.py --train-manifest data/manifests/seame_phaseII_train_manifest.csv --val-manifest data/manifests/seame_phaseII_val_manifest.csv --test-manifest data/manifests/seame_phaseII_test_manifest.csv --cuda --batch-size 8 --labels-path data/labels/hkust_16khz_aishell_cv_valid_labels.json --lr 1e-4 --name train_seame_drop0.1_cnn_batch8_layers3_heads8_lr1e-4_emb512 --save-folder save/ --save-every 1 --emb_cnn --dropout 0.1 --num-layers 3 --num-heads 8 --dim-model 512 --dim-key 64 --dim-value 64 --dim-input 161 --dim-inner 512 --dim-emb 512 --epochs 2000 --src-max-len 2000 --tgt-max-len 1000
```

```
CUDA_VISIBLE_DEVICES=2 python3 train.py --train-manifest data/manifests/seame_phaseII_train_manifest.csv --val-manifest data/manifests/seame_phaseII_val_manifest.csv --test-manifest data/manifests/seame_phaseII_test_manifest.csv --cuda --batch-size 8 --labels-path data/labels/hkust_16khz_aishell_cv_valid_labels.json --lr 1e-4 --name train_seame_drop0.2_cnn_batch8_layers3_heads8_lr1e-4 --save-folder save/ --save-every 1 --emb_cnn --dropout 0.2 --num-layers 3 --num-heads 8 --dim-model 256 --dim-key 64 --dim-value 64 --dim-input 161 --dim-inner 512 --dim-emb 256 --epochs 2000 --src-max-len 2000 --tgt-max-len 1000
```

```
CUDA_VISIBLE_DEVICES=3 python3 train.py --train-manifest data/manifests/seame_phaseII_train_manifest.csv --val-manifest data/manifests/seame_phaseII_val_manifest.csv --test-manifest data/manifests/seame_phaseII_test_manifest.csv --cuda --batch-size 8 --labels-path data/labels/hkust_16khz_aishell_cv_valid_labels.json --lr 1e-4 --name train_seame_drop0.3_cnn_batch8_layers3_heads8_lr1e-4 --save-folder save/ --save-every 1 --emb_cnn --dropout 0.3 --num-layers 3 --num-heads 8 --dim-model 256 --dim-key 64 --dim-value 64 --dim-input 161 --dim-inner 512 --dim-emb 256 --epochs 2000 --src-max-len 2000 --tgt-max-len 1000
```

LAS
```
CUDA_VISIBLE_DEVICES=3 python3 train.py --model=LAS --train-manifest data/manifests/seame_phaseII_train_manifest.csv --val-manifest data/manifests/seame_phaseII_val_manifest.csv --test-manifest data/manifests/seame_phaseII_test_manifest.csv --cuda --batch-size 8 --labels-path data/labels/hkust_16khz_aishell_cv_valid_labels.json --lr 1e-4 --name train_las_drop0.1_cnn_batch8_layers3_heads8_lr1e-4 --save-folder save/ --save-every 1
```

## Test
```
CUDA_VISIBLE_DEVICES=4 python3 test.py --test-manifest data/manifests/cv-valid-test_manifest.csv --continue-from save/multi_train_drop0.1_cnn_batch16_layers5_heads8/epoch_7.th --verbose --dropout 0.1 --num-layers 5 --num-heads 8 --dim-model 512 --dim-key 64 --dim-value 64 --dim-input 161 --dim-inner 1024 --dim-emb 512 --epochs 2000 --src-max-len 2000 --tgt-max-len 1000
```

```
CUDA_VISIBLE_DEVICES=3 python3 test.py --test-manifest data/manifests/hkust_16khz_val_manifest.csv --continue-from save/multi_train_drop0.1_cnn_batch8_layers3_heads8_lr1e-4/epoch_6.th --verbose --cuda --batch-size 1
```