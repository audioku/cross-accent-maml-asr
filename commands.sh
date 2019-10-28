# Libri Small Model
CUDA_VISIBLE_DEVICES=2 python3 train.py --train-manifest data/manifests/libri_train_manifest.csv --val-manifest data/manifests/libri_val_manifest.csv --test-manifest data/manifests/libri_test_clean_manifest.csv --cuda --batch-size 8 --labels-path labels.json --lr 1e-4 --name libri_drop0.1_cnn_batch12_vgg_labelsmoothing0.1_layer4 --save-folder save/ --save-every 5 --vgg_cnn --dropout 0.1 --num-layers 4 --num-heads 8 --dim-model 512 --dim-key 64 --dim-value 64 --dim-input 5120 --dim-inner 2048 --dim-emb 512 --shuffle --label-smoothing 0.1

# Libri Large Model
CUDA_VISIBLE_DEVICES=1 python3 train.py --train-manifest data/manifests/libri_train_manifest.csv --val-manifest data/manifests/libri_val_manifest.csv --test-manifest data/manifests/libri_test_clean_manifest.csv --cuda --batch-size 8 --labels-path labels.json --lr 1e-4 --name libri_train_drop0.1_cnn_batch8_large_emb_sharing --save-folder save/ --save-every 1 --emb_cnn --dropout 0.1 --num-layers 6 --num-heads 8 --dim-model 512 --dim-key 64 --dim-value 64 --dim-input 161 --dim-inner 2048 --dim-emb 512 --emb_trg_sharing





CUDA_VISIBLE_DEVICES=2 python3 train.py --train-manifest data/manifests/aishell_train_manifest.csv --val-manifest data/manifests/aishell_dev_manifest.csv --test-manifest data/manifests/aishell_test_manifest.csv --cuda --batch-size 8 --labels-path data/labels/aishell_labels.json --lr 1e-4 --name aishell_drop0.1_cnn_batch8 --save-folder save/ --save-every 1 --emb_cnn --dropout 0.1 --num-layers 3 --num-heads 5 --dim-model 512 --dim-key 64 --dim-value 64 --dim-input 161 --dim-inner 1024 --dim-emb 512 --shuffle



CUDA_VISIBLE_DEVICES=1 python3 train.py --train-manifest data/manifests/aishell_train_manifest.csv --val-manifest data/manifests/aishell_dev_manifest.csv --test-manifest data/manifests/aishell_test_manifest.csv --cuda --batch-size 12 --labels-path data/labels/aishell_labels.json --lr 3e-4 --name aishell_drop0.1_cnn_batch12 --save-folder save/ --save-every 5 --emb_cnn --dropout 0.1 --num-layers 4 --num-heads 5 --dim-model 512 --dim-key 64 --dim-value 64 --dim-input 161 --dim-inner 512 --dim-emb 512 --shuffle



CUDA_VISIBLE_DEVICES=0 python3 train.py --train-manifest data/manifests/aishell_train_manifest.csv --val-manifest data/manifests/aishell_dev_manifest.csv --test-manifest data/manifests/aishell_test_manifest.csv --cuda --batch-size 12 --labels-path data/labels/aishell_labels.json --lr 1e-2 --name aishell_drop0.1_cnn_batch12_lr1e-2 --save-folder save/ --save-every 5 --emb_cnn --dropout 0.1 --num-layers 4 --num-heads 5 --dim-model 512 --dim-key 64 --dim-value 64 --dim-input 161 --dim-inner 512 --dim-emb 512 --shuffle



CUDA_VISIBLE_DEVICES=2 python3 train.py --train-manifest data/manifests/aishell_train_manifest.csv --val-manifest data/manifests/aishell_dev_manifest.csv --test-manifest data/manifests/aishell_test_manifest.csv --cuda --batch-size 12 --labels-path data/labels/aishell_labels.json --lr 3e-4 --name aishell_drop0.1_cnn_batch12_3 --save-folder save/ --save-every 5 --emb_cnn --dropout 0.1 --num-layers 2 --num-heads 8 --dim-model 256 --dim-key 64 --dim-value 64 --dim-input 161 --dim-inner 256 --dim-emb 256 --shuffle

CUDA_VISIBLE_DEVICES=3 python3 train.py --train-manifest data/manifests/aishell_train_manifest.csv --val-manifest data/manifests/aishell_dev_manifest.csv --test-manifest data/manifests/aishell_test_manifest.csv --cuda --batch-size 12 --labels-path data/labels/aishell_labels.json --lr 3e-4 --name aishell_drop0.1_cnn_batch12_4 --save-folder save/ --save-every 5 --emb_cnn --dropout 0.1 --num-layers 6 --num-heads 8 --dim-model 512 --dim-key 64 --dim-value 64 --dim-input 161 --dim-inner 2048 --dim-emb 512 --shuffle





CUDA_VISIBLE_DEVICES=3 python3 train.py --train-manifest data/manifests/aishell_train_manifest.csv --val-manifest data/manifests/aishell_dev_manifest.csv --test-manifest data/manifests/aishell_test_manifest.csv --cuda --batch-size 12 --labels-path data/labels/aishell_labels.json --lr 3e-4 --name aishell_drop0.1_cnn_batch12_4_vgg --save-folder save/ --save-every 5 --vgg_cnn --dropout 0.1 --num-layers 6 --num-heads 8 --dim-model 512 --dim-key 64 --dim-value 64 --dim-input 161 --dim-inner 2048 --dim-emb 512 --shuffle

CUDA_VISIBLE_DEVICES=2 python3 train.py --train-manifest data/manifests/aishell_train_manifest.csv --val-manifest data/manifests/aishell_dev_manifest.csv --test-manifest data/manifests/aishell_test_manifest.csv --cuda --batch-size 12 --labels-path data/labels/aishell_labels.json --lr 3e-4 --name aishell_drop0.1_cnn_batch12_4_vgg2 --save-folder save/ --save-every 5 --vgg_cnn --dropout 0.1 --num-layers 6 --num-heads 8 --dim-model 512 --dim-key 64 --dim-value 64 --dim-input 161 --dim-inner 2048 --dim-emb 512 --shuffle

CUDA_VISIBLE_DEVICES=3 python3 train.py --train-manifest data/manifests/aishell_train_manifest.csv --val-manifest data/manifests/aishell_dev_manifest.csv --test-manifest data/manifests/aishell_test_manifest.csv --cuda --batch-size 12 --labels-path data/labels/aishell_labels.json --lr 3e-4 --name aishell_drop0.1_cnn_batch12_vgg_labelsmoothing0.1_layer4 --save-folder save/ --save-every 5 --vgg_cnn --dropout 0.1 --num-layers 4 --num-heads 8 --dim-model 512 --dim-key 64 --dim-value 64 --dim-input 5120 --dim-inner 2048 --dim-emb 512 --shuffle --label-smoothing 0.1

CUDA_VISIBLE_DEVICES=2 python3 train.py --train-manifest data/manifests/aishell_train_manifest.csv --val-manifest data/manifests/aishell_dev_manifest.csv --test-manifest data/manifests/aishell_test_manifest.csv --cuda --batch-size 12 --labels-path data/labels/aishell_labels.json --lr 3e-4 --name aishell_drop0.1_cnn_batch12_4_vgg3 --save-folder save/ --save-every 5 --vgg_cnn --dropout 0.1 --num-layers 4 --num-heads 8 --dim-model 512 --dim-key 64 --dim-value 64 --dim-input 5120 --dim-inner 2048 --dim-emb 512 --shuffle





CUDA_VISIBLE_DEVICES=3 python3 train.py --train-manifest data/manifests/aishell_train_manifest.csv --val-manifest data/manifests/aishell_dev_manifest.csv --test-manifest data/manifests/aishell_test_manifest.csv --cuda --batch-size 12 --labels-path data/labels/aishell_labels.json --name aishell_drop0.1_cnn_batch12_vgg_layer6_head8_minlr-1e-5_klr-1 --save-folder save/ --save-every 5 --vgg_cnn --dropout 0.1 --num-layers 6 --num-heads 8 --dim-model 512 --dim-key 64 --dim-value 64 --dim-input 5120 --dim-inner 2048 --dim-emb 512 --shuffle --min-lr 1e-5 --k-lr 1






CUDA_VISIBLE_DEVICES=0 python3 train.py --train-manifest data/manifests/aishell_train_manifest.csv --val-manifest data/manifests/aishell_dev_manifest.csv --test-manifest data/manifests/aishell_test_manifest.csv --cuda --batch-size 12 --labels-path data/labels/aishell_labels.json --lr 1e-4 --name aishell_drop0.1_cnn_batch12_4_vgg_fp16_layer4 --save-folder save/ --save-every 5 --vgg_cnn --dropout 0.1 --num-layers 4 --num-heads 8 --dim-model 512 --dim-key 64 --dim-value 64 --dim-input 161 --dim-inner 2048 --dim-emb 512 --shuffle



CUDA_VISIBLE_DEVICES=3 python3 train.py --train-manifest-list data/manifests/aishell_train_manifest.csv --valid-manifest-list data/manifests/aishell_dev_manifest.csv --test-manifest-list data/manifests/aishell_test_manifest.csv --cuda --batch-size 12 --labels-path data/labels/aishell_labels.json --name aishell_drop0.1_cnn_batch12_vgg_layer6_head8_minlr-1e-5_klr-1_label0.1_sos_eos --save-folder save/ --save-every 5 --feat_extractor vgg_cnn --dropout 0.1 --num-layers 6 --num-heads 8 --dim-model 512 --dim-key 64 --dim-value 64 --dim-input 5120 --dim-inner 2048 --dim-emb 512 --shuffle --min-lr 1e-5 --k-lr 1 --label-smoothing 0.1