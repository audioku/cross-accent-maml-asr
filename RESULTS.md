TEST CER:29.83% WER:52.64%: 100%|███████████| 1164/1164 [07:22<00:00,  2.62it/s]
[1]+  Terminated              CUDA_VISIBLE_DEVICES=3 python3 test.py --cuda --verbose --batch-size 1 --continue-from save/train_seame_16khz_drop0.1_cnn_batch8_layers3_heads8_lr1e-4_shuffle/epoch_49.th --test-manifest data/manifests/seame_phaseII_test_manifest.csv
(end2end_asr) genta@black-cube-1:~/end2end_asr$ CUDA_VISIBLE_DEVICES=3 python3 test.py --cuda --verbose --batch-size 4 --continue-from save/train_seame_16khz_drop0.1_cnn_batch8_layers3_heads8_lr1e-4_shuffle/epoch_49.th --test-manifest data/manifests/seame_phaseII_test_manifest.csv

TEST CER:29.30% WER:51.61%: 100%|███████████| 1164/1164 [10:48<00:00,  1.53it/s]
(end2end_asr) genta@black-cube-1:~/end2end_asr$ CUDA_VISIBLE_DEVICES=3 python3 test.py --cuda --verbose --batch-size 4 --test-manifest data/manifests/seame_phaseII_test_manifest.csv --continue-from save/train_seame_16khz_drop0.1_cnn_batch8_layers3_heads8_lr1e-4_shuffle/epoch_72.th

TEST CER:29.03% WER:50.85%: 100%|███████████| 1164/1164 [08:04<00:00,  2.49it/s]
(end2end_asr) genta@black-cube-1:~/end2end_asr$ CUDA_VISIBLE_DEVICES=0 python3 test.py --cuda --batch-size 4 --verbose --test-manifest data/manifests/seame_phaseII_test_manifest.csv --continue-from save/finetune_train_seame_16khz_drop0.1_cnn_batch8_layers3_heads8_lr1e-4_shuffle/epoch_24.th

TEST CER:28.34% WER:50.08%: 100%|███████████| 1164/1164 [07:50<00:00,  2.54it/s]
(end2end_asr) genta@black-cube-1:~/end2end_asr$ CUDA_VISIBLE_DEVICES=0 python3 test.py --cuda --batch-size 4 --verbose --test-manifest data/manifests/seame_phaseII_test_manifest.csv --continue-from save/finetune_train_seame_16khz_drop0.1_cnn_batch8_layers3_heads8_lr1e-4_shuffle/epoch_25.th

TEST CER:28.57% WER:50.45%: 100%|███████████| 1164/1164 [07:49<00:00,  2.43it/s]
(end2end_asr) genta@black-cube-1:~/end2end_asr$ CUDA_VISIBLE_DEVICES=0 python3 test.py --cuda --batch-size 4 --verbose --test-manifest data/manifests/seame_phaseII_test_manifest.csv --continue-from save/finetune_train_seame_16khz_drop0.1_cnn_batch8_layers3_heads8_lr1e-4_shuffle/epoch_26.th

TEST CER:27.80% WER:49.28%: 100%|███████████| 1164/1164 [07:51<00:00,  2.60it/s]
(end2end_asr) genta@black-cube-1:~/end2end_asr$ CUDA_VISIBLE_DEVICES=0 python3 test.py --cuda --batch-size 4 --verbose --test-manifest data/manifests/seame_phaseII_test_manifest.csv --continue-from save/finetune_train_seame_16khz_drop0.1_cnn_batch8_layers3_heads8_lr1e-4_shuffle/epoch_28.th

TEST CER:27.62% WER:49.25%: 100%|███████████| 1164/1164 [07:20<00:00,  2.65it/s]
(end2end_asr) genta@black-cube-1:~/end2end_asr$ CUDA_VISIBLE_DEVICES=0 python3 test.py --cuda --verbose --batch-size 4 --test-manifest data/manifests/seame_phaseII_test_manifest.csv --continue-from save/finetune_train_seame_16khz_drop0.1_cnn_batch8_layers3_heads8_lr1e-4_shuffle/epoch_32.th

TEST CER:26.98% WER:48.41%: 100%|███████████| 1164/1164 [07:17<00:00,  2.35it/s]
(end2end_asr) genta@black-cube-1:~/end2end_asr$ CUDA_VISIBLE_DEVICES=0 python3 test.py --cuda --verbose --batch-size 4 --test-manifest data/manifests/seame_phaseII_test_manifest.csv --continue-from save/finetune_train_seame_16khz_drop0.1_cnn_batch8_layers3_heads8_lr1e-4_shuffle/epoch_36.th

TEST CER:26.64% WER:47.65%: 100%|███████████| 1164/1164 [09:51<00:00,  2.24it/s]
(end2end_asr) genta@black-cube-1:~/end2end_asr$ CUDA_VISIBLE_DEVICES=0 python3 test.py --cuda --verbose --batch-size 4 --test-manifest data/manifests/seame_phaseII_test_manifest.csv --continue-from save/finetune_train_seame_16khz_drop0.1_cnn_batch8_layers3_heads8_lr1e-4_shuffle/epoch_40.th




TEST CER:27.33% WER:49.09%: 100%|█████████| 1164/1164 [1:42:52<00:00,  5.99s/it]
(end2end_asr) genta@black-cube-1:~/end2end_asr$ CUDA_VISIBLE_DEVICES=3 python3 test.py --cuda --verbose --batch-size 4 --test-manifest data/manifests/seame_phaseII_test_manifest.csv --continue-from save/train_seame_16khz_drop0.1_cnn_batch8_layers3_heads8_lr1e-4_shuffle/epoch_72.th --beam-search --beam-nbest 1

TEST CER:26.13% WER:46.69%: 100%|█████████| 4654/4654 [1:21:14<00:00,  2.10s/it]
[1]+  Terminated              CUDA_VISIBLE_DEVICES=1 python3 test.py --cuda --verbose --batch-size 1 --test-manifest data/manifests/seame_phaseII_test_manifest.csv --continue-from save/finetune_train_seame_16khz_drop0.1_cnn_batch8_layers3_heads8_lr1e-4_shuffle/epoch_40.th --beam-search --beam-nbest 1
(end2end_asr) genta@black-cube-1:~/end2end_asr$

TEST CER:27.55% WER:48.26%: 100%|████████| 4654/4654 [21:24:29<00:00, 10.51s/it]
CUDA_VISIBLE_DEVICES=2 python3 test.py --cuda --verbose --batch-size 1 --test-manifest data/manifests/seame_phaseII_test_manifest.csv --continue-from save/finetune_train_seame_16khz_drop0.1_cnn_batch8_layers3_heads8_lr1e-4_shuffle/epoch_40.th --beam-width 8 --beam-search --beam-nbest 1 --c-weight 0


CUDA_VISIBLE_DEVICES=0 python3 test.py --cuda --verbose --batch-size 1 --test-manifest data/manifests/seame_phaseII_test_manifest.csv --continue-from save/finetune_train_seame_16khz_drop0.1_cnn_batch8_layers3_heads8_lr1e-4_shuffle/epoch_40.th --beam-width 8 --beam-search --beam-nbest 1 --c-weight 0.5



 CUDA_VISIBLE_DEVICES=3 python3 test.py --cuda --verbose --batch-size 1 --test-manifest data/manifests/seame_phaseII_test_manifest.csv --continue-from save/finetune_train_seame_16khz_drop0.1_cnn_batch8_layers3_heads8_lr1e-4_shuffle/epoch_40.th --beam-search --beam-nbest 1 --lm-rescoring --lm-path awd_lstm_lm/params_SEAME_batch32_dropi0.6_droph0.4.pt --lm-weight 0.4 --beam-width 4 --c-weight 1

  CUDA_VISIBLE_DEVICES=1 python3 test.py --cuda --verbose --batch-size 1 --test-manifest data/manifests/seame_phaseII_test_manifest.csv --continue-from save/finetune_train_seame_16khz_drop0.1_cnn_batch8_layers3_heads8_lr1e-4_shuffle/epoch_40.th --beam-search --beam-nbest 1 --lm-rescoring --lm-path awd_lstm_lm/params_SEAME_batch32_dropi0.6_droph0.4.pt --lm-weight 1 --beam-width 4 --c-weight 1

 CUDA_VISIBLE_DEVICES=0 python3 test.py --cuda --verbose --batch-size 1 --test-manifest data/manifests/seame_phaseII_test_manifest.csv --continue-from save/finetune_train_seame_16khz_drop0.1_cnn_batch8_layers3_heads8_lr1e-4_shuffle/epoch_40.th --beam-search --beam-nbest 1 --beam-width 4 --c-weight 1


CUDA_VISIBLE_DEVICES=2 python3 test.py --cuda --verbose --batch-size 1 --test-manifest data/manifests/seame_phaseII_test_manifest.csv --continue-from save/finetune_train_seame_16khz_drop0.1_cnn_batch8_layers3_heads8_lr1e-4_shuffle/epoch_40.th --beam-width 32 --beam-search --beam-nbest 1 --c-weight 0

CUDA_VISIBLE_DEVICES=3 python3 test.py --cuda --verbose --batch-size 1 --test-manifest data/manifests/seame_phaseII_test_manifest.csv --continue-from save/finetune_train_seame_16khz_drop0.1_cnn_batch8_layers3_heads8_lr1e-4_shuffle/epoch_40.th --beam-width 16 --beam-search --beam-nbest 1 --c-weight 0.5







NEW RESULTS

TEST CER:25.34% WER:46.12%: 100%|████████████████████████████████████████| 4654/4654 [2:17:10<00:00,  2.49s/it]
(end2end_asr) genta@black-cube-1:~/end2end_asr$ CUDA_VISIBLE_DEVICES=3 python3 test.py --cuda --verbose --batch-size 1 --test-manifest data/manifests/seame_phaseII_test_manifest.csv --continue-from save/finetune_train_seame_16khz_drop0.1_cnn_batch8_layers3_heads8_lr1e-4_shuffle/epoch_40.th --beam-width 4 --beam-search --beam-nbest 1 --c-weight 0.5


TEST CER:25.08% WER:45.92%: 100%|████████████████████████████████████████| 4654/4654 [7:41:45<00:00, 10.26s/it]
(end2end_asr) genta@black-cube-1:~/end2end_asr$
(end2end_asr) genta@black-cube-1:~/end2end_asr$ CUDA_VISIBLE_DEVICES=2 python3 test.py --cuda --verbose --batch-size 1 --test-manifest data/manifests/seame_phaseII_test_manifest.csv --continue-from save/finetune_train_seame_16khz_drop0.1_cnn_batch8_layers3_heads8_lr1e-4_shuffle/epoch_40.th --beam-width 8 --beam-search --beam-nbest 1 --c-weight 0.5

TEST CER:27.55% WER:48.26%: 100%|████████████████████████████████████████████████████████████████| 4654/4654 [7:48:51<00:00, 10.47s/it]
(end2end_asr) genta@black-cube-1:~/end2end_asr$ CUDA_VISIBLE_DEVICES=1 python3 test.py --cuda --verbose --batch-size 1 --test-manifest data/manifests/seame_phaseII_test_manifest.csv --continue-from save/finetune_train_seame_16khz_drop0.1_cnn_batch8_layers3_heads8_lr1e-4_shuffle/epoch_40.th --beam-width 8 --beam-search --beam-nbest 1 --c-weight 0

Rescoring
CUDA_VISIBLE_DEVICES=3 python3 test.py --cuda --verbose --batch-size 1 --test-manifest data/manifests/seame_phaseII_test_manifest.csv --continue-from save/finetune_train_seame_16khz_drop0.1_cnn_batch8_layers3_heads8_lr1e-4_shuffle/epoch_40.th --beam-width 8 --beam-search --beam-nbest 1 --c-weight 0.5 --lm-rescoring --lm-path multitask_lm/convert_finetune_train_nbest3_bsz64_drop0.2_emsize200_nhid200_lr20_modelLSTM_bptt35_lr20.0_drop0.2_layers2_nhid200_emsize200.txt --lm-weight 0.5

CUDA_VISIBLE_DEVICES=1 python3 test.py --cuda --verbose --batch-size 1 --test-manifest data/manifests/seame_phaseII_test_manifest.csv --continue-from save/finetune_train_seame_16khz_drop0.1_cnn_batch8_layers3_heads8_lr1e-4_shuffle/epoch_40.th --beam-width 8 --beam-search --beam-nbest 1 --c-weight 0.5 --lm-rescoring --lm-path multitask_lm/convert_train_bsz64_drop0.4_emsize200_nhid200_modelLSTM_bptt35_lr20_drop0.4_layers2_nhid200_emsize200.txt --lm-weight 0.25


parser.add_argument('--lm-rescoring', action='store_true', help='Rescore using LM')
parser.add_argument('--lm-path', type=str, default="lm_model.pt", help="Path to LM model")
parser.add_argument('--lm-weight', default=0.1, type=float, help='LM weight')












TEST CER:27.76% WER:51.06%: 100%|█████████| 4654/4654 [2:18:41<00:00,  2.01s/it]
(end2end_asr) genta@black-cube-1:~/end2end_asr$ CUDA_VISIBLE_DEVICES=0 python3 test.py --cuda --verbose --batch-size 1 --test-manifest data/manifests/seame_phaseII_test_manifest.csv --continue-from save/train_seame_16khz_drop0.1_cnn_batch8_layers3_heads8_lr1e-4_shuffle/epoch_72.th --beam-width 4 --beam-search --beam-nbest 1 --c-weight 3

CUDA_VISIBLE_DEVICES=0 python3 test.py --cuda --verbose --batch-size 1 --test-manifest data/manifests/seame_phaseII_test_manifest.csv --continue-from save/finetune_train_seame_16khz_drop0.1_cnn_batch8_layers3_heads8_lr1e-4_shuffle/epoch_40.th --beam-width 4 --beam-search --beam-nbest 1 --c-weight 3 --lm-rescoring --lm-path multitask_lm/convert_finetune_train_nbest3_bsz64_drop0.2_emsize200_nhid200_lr20_modelLSTM_bptt35_lr20.0_drop0.2_layers2_nhid200_emsize200.txt --lm-weight 0.52

TEST CER:26.36% WER:46.92% (without LM)
CUDA_VISIBLE_DEVICES=1 python3 test.py --cuda --verbose --batch-size 1 --test-manifest data/manifests/seame_phaseII_test_manifest.csv --continue-from save/finetune_train_seame_16khz_drop0.1_cnn_batch8_layers3_heads8_lr1e-4_shuffle/epoch_40.th --beam-width 12 --beam-search --beam-nbest 1 --c-weight 1

CUDA_VISIBLE_DEVICES=1 python3 test.py --cuda --verbose --batch-size 1 --test-manifest data/manifests/seame_phaseII_test_manifest.csv --continue-from save/finetune_train_seame_16khz_drop0.1_cnn_batch8_layers3_heads8_lr1e-4_shuffle/epoch_40.th --beam-width 8 --beam-search --beam-nbest 1 --c-weight 1

TEST CER:25.29% WER:45.76%: 100%|█████████| 4654/4654 [2:22:55<00:00,  2.80s/it]
(end2end_asr) genta@black-cube-1:~/end2end_asr$ CUDA_VISIBLE_DEVICES=2 python3 test.py --cuda --verbose --batch-size 1 --test-manifest data/manifests/seame_phaseII_test_manifest.csv --continue-from save/finetune_train_seame_16khz_drop0.1_cnn_batch8_layers3_heads8_lr1e-4_shuffle/epoch_40.th --beam-width 4 --beam-search --beam-nbest 1 --c-weight 3 --lm-rescoring --lm-path multitask_lm/convert_train_bsz64_drop0.4_emsize200_nhid200_modelLSTM_bptt35_lr20_drop0.4_layers2_nhid200_emsize200.txt --lm-weight 0.52


TEST CER:25.28% WER:45.79%: 100%|█████████| 4654/4654 [2:22:05<00:00,  2.68s/it]
(end2end_asr) genta@black-cube-1:~/end2end_asr$ CUDA_VISIBLE_DEVICES=0 python3 test.py --cuda --verbose --batch-size 1 --test-manifest data/manifests/seame_phaseII_test_manifest.csv --continue-from save/finetune_train_seame_16khz_drop0.1_cnn_batch8_layers3_heads8_lr1e-4_shuffle/epoch_40.th --beam-width 4 --beam-search --beam-nbest 1 --c-weight 3 --lm-rescoring --lm-path multitask_lm/convert_finetune_train_nbest3_bsz64_drop0.2_emsize200_nhid200_lr20_modelLSTM_bptt35_lr20.0_drop0.2_layers2_nhid200_emsize200.txt --lm-weight 0.22



25.26% WER:45.68%
CUDA_VISIBLE_DEVICES=3 python3 test.py --cuda --verbose --batch-size 1 --test-manifest data/manifests/seame_phaseII_test_manifest.csv --continue-from save/finetune_train_seame_16khz_drop0.1_cnn_batch8_layers3_heads8_lr1e-4_shuffle/epoch_40.th --beam-width 4 --beam-search --beam-nbest 1 --c-weight 3 --lm-rescoring --lm-path multitask_lm/convert_finetune_train_nbest3_bsz64_drop0.2_emsize200_nhid200_lr20_modelLSTM_bptt35_lr20.0_drop0.2_layers2_nhid200_emsize200.txt --lm-weight 0.52

TEST CER:25.29% WER:45.72%: 100%|█████████| 4654/4654 [2:23:37<00:00,  2.79s/it]
(end2end_asr) genta@black-cube-1:~/end2end_asr$ CUDA_VISIBLE_DEVICES=0 python3 test.py --cuda --verbose --batch-size 1 --test-manifest data/manifests/seame_phaseII_test_manifest.csv --continue-from save/finetune_train_seame_16khz_drop0.1_cnn_batch8_layers3_heads8_lr1e-4_shuffle/epoch_40.th --beam-width 4 --beam-search --beam-nbest 1 --c-weight 3 --lm-rescoring --lm-path multitask_lm/convert_finetune_train_nbest3_bsz64_drop0.2_emsize200_nhid200_lr20_modelLSTM_bptt35_lr20.0_drop0.2_layers2_nhid200_emsize200.txt --lm-weight 0.62

TEST CER:25.32% WER:45.68%: 100%|█████████| 4654/4654 [2:24:30<00:00,  2.81s/it]
(end2end_asr) genta@black-cube-1:~/end2end_asr$ CUDA_VISIBLE_DEVICES=3 python3 test.py --cuda --verbose --batch-size 1 --test-manifest data/manifests/seame_phaseII_test_manifest.csv --continue-from save/finetune_train_seame_16khz_drop0.1_cnn_batch8_layers3_heads8_lr1e-4_shuffle/epoch_40.th --beam-width 4 --beam-search --beam-nbest 1 --c-weight 3 --lm-rescoring --lm-path multitask_lm/convert_finetune_train_nbest3_bsz64_drop0.2_emsize200_nhid200_lr20_modelLSTM_bptt35_lr20.0_drop0.2_layers2_nhid200_emsize200.txt --lm-weight 0.72



CUDA_VISIBLE_DEVICES=0 python3 test.py --cuda --verbose --batch-size 1 --test-manifest data/manifests/seame_phaseII_test_manifest.csv --continue-from save/finetune_train_seame_16khz_drop0.1_cnn_batch8_layers3_heads8_lr1e-4_shuffle/epoch_40.th --beam-width 4 --beam-search --beam-nbest 1 --c-weight 3 --lm-rescoring --lm-path multitask_lm/convert_finetune_train_nbest3_bsz64_drop0.2_emsize200_nhid200_lr20_modelLSTM_bptt35_lr20.0_drop0.2_layers2_nhid200_emsize200.txt --lm-weight 0.62

CUDA_VISIBLE_DEVICES=3 python3 test.py --cuda --verbose --batch-size 1 --test-manifest data/manifests/seame_phaseII_test_manifest.csv --continue-from save/finetune_train_seame_16khz_drop0.1_cnn_batch8_layers3_heads8_lr1e-4_shuffle/epoch_40.th --beam-width 4 --beam-search --beam-nbest 1 --c-weight 3 --lm-rescoring --lm-path multitask_lm/convert_finetune_train_nbest3_bsz64_drop0.2_emsize200_nhid200_lr20_modelLSTM_bptt35_lr20.0_drop0.2_layers2_nhid200_emsize200.txt --lm-weight 0.72

TEST CER:25.52% WER:45.79%: 100%|█████████| 4654/4654 [2:25:11<00:00,  2.82s/it]
(end2end_asr) genta@black-cube-1:~/end2end_asr$ CUDA_VISIBLE_DEVICES=1 python3 test.py --cuda --verbose --batch-size 1 --test-manifest data/manifests/seame_phaseII_test_manifest.csv --continue-from save/finetune_train_seame_16khz_drop0.1_cnn_batch8_layers3_heads8_lr1e-4_shuffle/epoch_40.th --beam-width 4 --beam-search --beam-nbest 1 --c-weight 3 --lm-rescoring --lm-path multitask_lm/convert_finetune_train_nbest3_bsz64_drop0.2_emsize200_nhid200_lr20_modelLSTM_bptt35_lr20.0_drop0.2_layers2_nhid200_emsize200.txt --lm-weight 1




















TEST CER:31.71% WER:46.08% CER_EN:39.71% CER_ZH:30.87%: 100%|█| 4654/4654 [2:06:40<00:00,  2.46s/it]
(end2end_asr) genta@black-cube-1:~/end2end_asr$ CUDA_VISIBLE_DEVICES=3 python3 test.py --cuda --verbose --batch-size 1 --test-manifest data/manifests/seame_phaseII_test_manifest.csv --continue-from save/finetune_train_seame_16khz_drop0.1_cnn_batch8_layers3_heads8_lr1e-4_shuffle/epoch_40.th --beam-width 12 --beam-search --beam-nbest 1 --c-weight 1








CUDA_VISIBLE_DEVICES=0 python3 test.py --cuda --verbose --batch-size 1 --test-manifest data/manifests/seame_phaseII_test_manifest.csv --continue-from save/train_seame_16khz_drop0.1_cnn_batch8_layers3_heads8_lr1e-4_shuffle/epoch_72.th --beam-width 8 --beam-search --beam-nbest 1 --c-weight 1

CUDA_VISIBLE_DEVICES=1 python3 test.py --cuda --verbose --batch-size 1 --test-manifest data/manifests/seame_phaseII_test_manifest.csv --continue-from save/finetune_train_seame_16khz_drop0.1_cnn_batch8_layers3_heads8_lr1e-4_shuffle/epoch_40.th --beam-width 8 --beam-search --beam-nbest 1 --c-weight 1

CUDA_VISIBLE_DEVICES=2 python3 test.py --cuda --verbose --batch-size 1 --test-manifest data/manifests/seame_phaseII_test_manifest.csv --continue-from save/finetune_train_seame_16khz_drop0.1_cnn_batch8_layers3_heads8_lr1e-4_shuffle/epoch_40.th --beam-width 8 --beam-search --beam-nbest 1 --c-weight 1.7 --lm-rescoring --lm-path multitask_lm/convert_train_bsz64_drop0.4_emsize200_nhid200_modelLSTM_bptt35_lr20_drop0.4_layers2_nhid200_emsize200.txt --lm-weight 0.5

CUDA_VISIBLE_DEVICES=3 python3 test.py --cuda --verbose --batch-size 1 --test-manifest data/manifests/seame_phaseII_test_manifest.csv --continue-from save/finetune_train_seame_16khz_drop0.1_cnn_batch8_layers3_heads8_lr1e-4_shuffle/epoch_40.th --beam-width 8 --beam-search --beam-nbest 1 --c-weight 2 --lm-rescoring --lm-path multitask_lm/convert_finetune_train_nbest3_bsz64_drop0.2_emsize200_nhid200_lr20_modelLSTM_bptt35_lr20.0_drop0.2_layers2_nhid200_emsize200.txt --lm-weight 0.5



TEST CER:31.75% WER:45.89% CER_EN:39.03% CER_ZH:31.37%: 100%|██████| 4654/4654 [7:26:45<00:00,  9.00s/it]
CUDA_VISIBLE_DEVICES=3 python3 test.py --cuda --verbose --batch-size 1 --test-manifest data/manifests/seame_phaseII_test_manifest.csv --continue-from save/finetune_train_seame_16khz_drop0.1_cnn_batch8_layers3_heads8_lr1e-4_shuffle/epoch_40.th --beam-width 8 --beam-search --beam-nbest 1 --c-weight 2 --lm-rescoring --lm-path multitask_lm/convert_finetune_train_nbest3_bsz64_drop0.2_emsize200_nhid200_lr20_modelLSTM_bptt35_lr20.0_drop0.2_layers2_nhid200_emsize200.txt --lm-weight 0.6

TEST CER:32.25% WER:46.22% CER_EN:39.45% CER_ZH:31.90%: 100%|█| 4654/4654 [7:25:04<00:00,  9.24s/it]
(end2end_asr) genta@black-cube-1:~/end2end_asr$ CUDA_VISIBLE_DEVICES=2 python3 test.py --cuda --verbose --batch-size 1 --test-manifest data/manifests/seame_phaseII_test_manifest.csv --continue-from save/finetune_train_seame_16khz_drop0.1_cnn_batch8_layers3_heads8_lr1e-4_shuffle/epoch_40.th --beam-width 8 --beam-search --beam-nbest 1 --c-weight 1.7 --lm-rescoring --lm-path multitask_lm/convert_train_bsz64_drop0.4_emsize200_nhid200_modelLSTM_bptt35_lr20_drop0.4_layers2_nhid200_emsize200.txt --lm-weight 0.5

TEST CER:32.76% WER:46.88% CER_EN:40.06% CER_ZH:32.44%: 100%|███████████████████| 4654/4654 [7:02:15<00:00,  8.87s/it]
(end2end_asr) genta@black-cube-1:~/end2end_asr$ CUDA_VISIBLE_DEVICES=1 python3 test.py --cuda --verbose --batch-size 1 --test-manifest data/manifests/seame_phaseII_test_manifest.csv --continue-from save/finetune_train_seame_16khz_drop0.1_cnn_batch8_layers3_heads8_lr1e-4_shuffle/epoch_40.th --beam-width 8 --beam-search --beam-nbest 1 --c-weight 1

TEST CER:34.40% WER:50.51% CER_EN:41.70% CER_ZH:35.84%: 100%|█| 4654/4654 [6:59:18<00:00,  8.82s/it]
(end2end_asr) genta@black-cube-1:~/end2end_asr$ CUDA_VISIBLE_DEVICES=0 python3 test.py --cuda --verbose --batch-size 1 --test-manifest data/manifests/seame_phaseII_test_manifest.csv --continue-from save/train_seame_16khz_drop0.1_cnn_batch8_layers3_heads8_lr1e-4_shuffle/epoch_72.th --beam-width 8 --beam-search --beam-nbest 1 --c-weight 1




CUDA_VISIBLE_DEVICES=0 python3 test.py --cuda --verbose --batch-size 1 --test-manifest data/manifests/seame_phaseII_test_manifest.csv --continue-from save/finetune_train_seame_16khz_drop0.1_cnn_batch8_layers3_heads8_lr1e-4_shuffle/epoch_40.th --beam-width 4 --beam-search --beam-nbest 1 --c-weight 2 --lm-rescoring --lm-path multitask_lm/convert_finetune_train_nbest3_bsz64_drop0.2_emsize200_nhid200_lr20_modelLSTM_bptt35_lr20.0_drop0.2_layers2_nhid200_emsize200.txt --lm-weight 0.6






TEST CER:31.55% WER:46.13% CER_EN:39.24% CER_ZH:31.05%: 100%|█| 4654/4654 [2:24:29<00:00,  2.84s/it]
(end2end_asr) genta@black-cube-1:~/end2end_asr$ CUDA_VISIBLE_DEVICES=0 python3 test.py --cuda --verbose --batch-size 1 --test-manifest data/manifests/seame_phaseII_test_manifest.csv --continue-from save/finetune_train_seame_16khz_drop0.1_cnn_batch8_layers3_heads8_lr1e-4_shuffle/epoch_40.th --beam-width 4 --beam-search --beam-nbest 1 --c-weight 2 --lm-rescoring --lm-path multitask_lm/convert_finetune_train_nbest3_bsz64_drop0.2_emsize200_nhid200_lr20_modelLSTM_bptt35_lr20.0_drop0.2_layers2_nhid200_emsize200.txt --lm-weight 0.4



CUDA_VISIBLE_DEVICES=1 python3 test.py --cuda --verbose --batch-size 1 --test-manifest data/manifests/seame_phaseII_test_manifest.csv --continue-from save/train_seame_16khz_drop0.1_cnn_batch8_layers3_heads8_lr1e-4_shuffle/epoch_72.th --beam-width 4 --beam-search --beam-nbest 1 --c-weight 1




TEST CER:34.40% WER:51.00% CER_EN:41.79% CER_ZH:35.94%: 100%|█| 4654/4654 [2:21:02<00:00,  1.94s/it]
(end2end_asr) genta@black-cube-1:~/end2end_asr$ CUDA_VISIBLE_DEVICES=1 python3 test.py --cuda --verbose --batch-size 1 --test-manifest data/manifests/seame_phaseII_test_manifest.csv --continue-from save/train_seame_16khz_drop0.1_cnn_batch8_layers3_heads8_lr1e-4_shuffle/epoch_72.th --beam-width 4 --beam-search --beam-nbest 1 --c-weight 1

TEST CER:31.42% WER:46.06% CER_EN:39.09% CER_ZH:30.91%: 100%|█| 4654/4654 [2:27:34<00:00,  2.90s/it]
(end2end_asr) genta@black-cube-1:~/end2end_asr$ CUDA_VISIBLE_DEVICES=0 python3 test.py --cuda --verbose --batch-size 1 --test-manifest data/manifests/seame_phaseII_test_manifest.csv --continue-from save/finetune_train_seame_16khz_drop0.1_cnn_batch8_layers3_heads8_lr1e-4_shuffle/epoch_40.th --beam-width 4 --beam-search --beam-nbest 1 --c-weight 2 --lm-rescoring --lm-path multitask_lm/convert_finetune_train_nbest3_bsz64_drop0.2_emsize200_nhid200_lr20_modelLSTM_bptt35_lr20.0_drop0.2_layers2_nhid200_emsize200.txt --lm-weight 0.3

TEST CER:31.91% WER:45.94% CER_EN:39.15% CER_ZH:31.58%: 100%|█| 4654/4654 [7:23:43<00:00, 10.56s/it]
(end2end_asr) genta@black-cube-1:~/end2end_asr$ CUDA_VISIBLE_DEVICES=2 python3 test.py --cuda --verbose --batch-size 1 --test-manifest data/manifests/seame_phaseII_test_manifest.csv --continue-from save/finetune_train_seame_16khz_drop0.1_cnn_batch8_layers3_heads8_lr1e-4_shuffle/epoch_40.th --beam-width 8 --beam-search --beam-nbest 1 --c-weight 2 --lm-rescoring --lm-path multitask_lm/convert_finetune_train_nbest3_bsz64_drop0.2_emsize200_nhid200_lr20_modelLSTM_bptt35_lr20.0_drop0.2_layers2_nhid200_emsize200.txt --lm-weight 0.6

TEST CER:31.31% WER:45.63% CER_EN:38.66% CER_ZH:30.93%: 100%|█| 4654/4654 [7:22:44<00:00, 10.70s/it]
(end2end_asr) genta@black-cube-1:~/end2end_asr$ CUDA_VISIBLE_DEVICES=3 python3 test.py --cuda --verbose --batch-size 1 --test-manifest data/manifests/seame_phaseII_test_manifest.csv --continue-from save/finetune_train_seame_16khz_drop0.1_cnn_batch8_layers3_heads8_lr1e-4_shuffle/epoch_40.th --beam-width 8 --beam-search --beam-nbest 1 --c-weight 2.5 --lm-rescoring --lm-path multitask_lm/convert_finetune_train_nbest3_bsz64_drop0.2_emsize200_nhid200_lr20_modelLSTM_bptt35_lr20.0_drop0.2_layers2_nhid200_emsize200.txt --lm-weight 0.6

TEST CER:31.19% WER:45.62% CER_EN:38.52% CER_ZH:30.91%: 100%|█| 4654/4654 [6:58:18<00:00,  8.96s/it]
(end2end_asr) genta@black-cube-1:~/end2end_asr$ CUDA_VISIBLE_DEVICES=2 python3 test.py --cuda --verbose --batch-size 1 --test-manifest data/manifests/seame_phaseII_test_manifest.csv --continue-from save/finetune_train_seame_16khz_drop0.1_cnn_batch8_layers3_heads8_lr1e-4_shuffle/epoch_40.th --beam-width 8 --beam-search --beam-nbest 1 --c-weight 2.8 --lm-rescoring --lm-path multitask_lm/convert_finetune_train_nbest3_bsz64_drop0.2_emsize200_nhid200_lr20_modelLSTM_bptt35_lr20.0_drop0.2_layers2_nhid200_emsize200.txt --lm-weight 0.6

TEST CER:31.07% WER:45.64% CER_EN:38.39% CER_ZH:30.85%: 100%|█| 4654/4654 [6:54:29<00:00,  9.18s/it]
(end2end_asr) genta@black-cube-1:~/end2end_asr$ CUDA_VISIBLE_DEVICES=3 python3 test.py --cuda --verbose --batch-size 1 --test-manifest data/manifests/seame_phaseII_test_manifest.csv --continue-from save/finetune_train_seame_16khz_drop0.1_cnn_batch8_layers3_heads8_lr1e-4_shuffle/epoch_40.th --beam-width 8 --beam-search --beam-nbest 1 --c-weight 3 --lm-rescoring --lm-path multitask_lm/convert_finetune_train_nbest3_bsz64_drop0.2_emsize200_nhid200_lr20_modelLSTM_bptt35_lr20.0_drop0.2_layers2_nhid200_emsize200.txt --lm-weight 0.6



CUDA_VISIBLE_DEVICES=3 python3 test.py --cuda --verbose --batch-size 1 --test-manifest data/manifests/seame_phaseII_test_manifest.csv --continue-from save/finetune_train_seame_16khz_drop0.1_cnn_batch8_layers3_heads8_lr1e-4_shuffle/epoch_40.th --beam-width 4 --beam-search --beam-nbest 1 --c-weight 3 --lm-rescoring --lm-path multitask_lm/convert_finetune_train_nbest3_bsz64_drop0.2_emsize200_nhid200_lr20_modelLSTM_bptt35_lr20.0_drop0.2_layers2_nhid200_emsize200.txt --lm-weight 0.6

CUDA_VISIBLE_DEVICES=2 python3 test.py --cuda --verbose --batch-size 1 --test-manifest data/manifests/seame_phaseII_test_manifest.csv --continue-from save/finetune_train_seame_16khz_drop0.1_cnn_batch8_layers3_heads8_lr1e-4_shuffle/epoch_40.th --beam-width 4 --beam-search --beam-nbest 1 --c-weight 3.5 --lm-rescoring --lm-path multitask_lm/convert_finetune_train_nbest3_bsz64_drop0.2_emsize200_nhid200_lr20_modelLSTM_bptt35_lr20.0_drop0.2_layers2_nhid200_emsize200.txt --lm-weight 0.6


TEST CER:31.45% WER:46.16% CER_EN:39.14% CER_ZH:30.93%: 100%|█| 4654/4654 [2:21:35<00:00,  2.71s/it]
(end2end_asr) genta@black-cube-1:~/end2end_asr$ CUDA_VISIBLE_DEVICES=3 python3 test.py --cuda --verbose --batch-size 1 --test-manifest data/manifests/seame_phaseII_test_manifest.csv --continue-from save/finetune_train_seame_16khz_drop0.1_cnn_batch8_layers3_heads8_lr1e-4_shuffle/epoch_40.th --beam-width 4 --beam-search --beam-nbest 1 --c-weight 3 --lm-rescoring --lm-path multitask_lm/convert_finetune_train_nbest3_bsz64_drop0.2_emsize200_nhid200_lr20_modelLSTM_bptt35_lr20.0_drop0.2_layers2_nhid200_emsize200.txt --lm-weight 0.6

TEST CER:31.42% WER:46.20% CER_EN:39.17% CER_ZH:30.92%: 100%|█| 4654/4654 [2:22:55<00:00,  2.64s/it]
(end2end_asr) genta@black-cube-1:~/end2end_asr$ CUDA_VISIBLE_DEVICES=2 python3 test.py --cuda --verbose --batch-size 1 --test-manifest data/manifests/seame_phaseII_test_manifest.csv --continue-from save/finetune_train_seame_16khz_drop0.1_cnn_batch8_layers3_heads8_lr1e-4_shuffle/epoch_40.th --beam-width 4 --beam-search --beam-nbest 1 --c-weight 3.5 --lm-rescoring --lm-path multitask_lm/convert_finetune_train_nbest3_bsz64_drop0.2_emsize200_nhid200_lr20_modelLSTM_bptt35_lr20.0_drop0.2_layers2_nhid200_emsize200.txt --lm-weight 0.6
