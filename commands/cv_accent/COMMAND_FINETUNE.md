##############################
# PHILIPPINES
##############################

###
# FINETUNE META 10%
###
CUDA_VISIBLE_DEVICES=1 python3 finetune.py \
--train-manifest-list ./data/manifests/cv_20190612_philippines_train.csv \
--valid-manifest-list ./data/manifests/cv_20190612_philippines_test.csv \
--test-manifest-list ./data/manifests/cv_20190612_philippines_test.csv \
--train-partition-list 0.1 \
--cuda --k-train 6 --labels-path data/labels/cv_labels.json --lr 1e-4 --name multi_accent_finetune_10shot_5updates_philippines_maml_10_3_3_enc2_dec4_512_b6_22050hz_copy_grad_early10000 --save-folder save/ --feat_extractor vgg_cnn --dropout 0.1 --num-enc-layers 2 --num-dec-layers 4 --num-heads 8 --dim-model 512 --dim-key 64 --dim-value 64 --dim-input 5120 --dim-inner 512 --dim-emb 512 --early-stop cer,50 --src-max-len 5000 --tgt-max-len 2500 --epochs 5 --sample-rate 22050 --continue-from save/maml_10_3_3_enc2_dec4_512_b6_22050hz_copy_grad_early10000/epoch_220000.th --beam-search --beam-width 5 --save-every 5 --opt_name sgd --evaluate-every 5 &

###
# FINETUNE META 25%
###
CUDA_VISIBLE_DEVICES=1 python3 finetune.py \
--train-manifest-list ./data/manifests/cv_20190612_philippines_train.csv \
--valid-manifest-list ./data/manifests/cv_20190612_philippines_test.csv \
--test-manifest-list ./data/manifests/cv_20190612_philippines_test.csv \
--train-partition-list 0.25 \
--cuda --k-train 6 --labels-path data/labels/cv_labels.json --lr 1e-4 --name multi_accent_finetune_25shot_5updates_philippines_maml_10_3_3_enc2_dec4_512_b6_22050hz_copy_grad_early10000 --save-folder save/ --feat_extractor vgg_cnn --dropout 0.1 --num-enc-layers 2 --num-dec-layers 4 --num-heads 8 --dim-model 512 --dim-key 64 --dim-value 64 --dim-input 5120 --dim-inner 512 --dim-emb 512 --early-stop cer,50 --src-max-len 5000 --tgt-max-len 2500 --epochs 5 --sample-rate 22050 --continue-from save/maml_10_3_3_enc2_dec4_512_b6_22050hz_copy_grad_early10000/epoch_220000.th --beam-search --beam-width 5 --save-every 5 --opt_name sgd --evaluate-every 5 &

###
# FINETUNE META 50%
###
CUDA_VISIBLE_DEVICES=1 python3 finetune.py \
--train-manifest-list ./data/manifests/cv_20190612_philippines_train.csv \
--valid-manifest-list ./data/manifests/cv_20190612_philippines_test.csv \
--test-manifest-list ./data/manifests/cv_20190612_philippines_test.csv \
--train-partition-list 0.5 \
--cuda --k-train 6 --labels-path data/labels/cv_labels.json --lr 1e-4 --name multi_accent_finetune_50shot_5updates_philippines_maml_10_3_3_enc2_dec4_512_b6_22050hz_copy_grad_early10000 --save-folder save/ --feat_extractor vgg_cnn --dropout 0.1 --num-enc-layers 2 --num-dec-layers 4 --num-heads 8 --dim-model 512 --dim-key 64 --dim-value 64 --dim-input 5120 --dim-inner 512 --dim-emb 512 --early-stop cer,50 --src-max-len 5000 --tgt-max-len 2500 --epochs 5 --sample-rate 22050 --continue-from save/maml_10_3_3_enc2_dec4_512_b6_22050hz_copy_grad_early10000/epoch_220000.th --beam-search --beam-width 5 --save-every 5 --opt_name sgd --evaluate-every 5 &

###
# FINETUNE META 100%
###
CUDA_VISIBLE_DEVICES=1 python3 finetune.py \
--train-manifest-list ./data/manifests/cv_20190612_philippines_train.csv \
--valid-manifest-list ./data/manifests/cv_20190612_philippines_test.csv \
--test-manifest-list ./data/manifests/cv_20190612_philippines_test.csv \
--train-partition-list 1 \
--cuda --k-train 6 --labels-path data/labels/cv_labels.json --lr 1e-4 --name multi_accent_finetune_100shot_5updates_philippines_maml_10_3_3_enc2_dec4_512_b6_22050hz_copy_grad_early10000 --save-folder save/ --feat_extractor vgg_cnn --dropout 0.1 --num-enc-layers 2 --num-dec-layers 4 --num-heads 8 --dim-model 512 --dim-key 64 --dim-value 64 --dim-input 5120 --dim-inner 512 --dim-emb 512 --early-stop cer,50 --src-max-len 5000 --tgt-max-len 2500 --epochs 5 --sample-rate 22050 --continue-from save/maml_10_3_3_enc2_dec4_512_b6_22050hz_copy_grad_early10000/epoch_220000.th --beam-search --beam-width 5 --save-every 5 --opt_name sgd --evaluate-every 5 &

###
# FINETUNE JOINT 10%
###
CUDA_VISIBLE_DEVICES=2 python3 finetune.py \
--train-manifest-list ./data/manifests/cv_20190612_philippines_train.csv \
--valid-manifest-list ./data/manifests/cv_20190612_philippines_test.csv \
--test-manifest-list ./data/manifests/cv_20190612_philippines_test.csv \
--train-partition-list 0.1 \
--cuda --k-train 6 --labels-path data/labels/cv_labels.json --lr 1e-4 --name multi_accent_finetune_10shot_5updates_philippines_joint_10_3_3_enc2_dec4_512_b6_22050hz --save-folder save/ --feat_extractor vgg_cnn --dropout 0.1 --num-enc-layers 2 --num-dec-layers 4 --num-heads 8 --dim-model 512 --dim-key 64 --dim-value 64 --dim-input 5120 --dim-inner 512 --dim-emb 512 --early-stop cer,50 --src-max-len 5000 --tgt-max-len 2500 --epochs 5 --sample-rate 22050 --continue-from save/joint_10_3_3_enc2_dec4_512_b6_22050hz/epoch_220000.th --beam-search --beam-width 5 --save-every 5 --opt_name sgd --evaluate-every 5 --training-mode joint &

###
# FINETUNE JOINT 25%
###
CUDA_VISIBLE_DEVICES=0 python3 finetune.py \
--train-manifest-list ./data/manifests/cv_20190612_philippines_train.csv \
--valid-manifest-list ./data/manifests/cv_20190612_philippines_test.csv \
--test-manifest-list ./data/manifests/cv_20190612_philippines_test.csv \
--train-partition-list 0.25 \
--cuda --k-train 6 --labels-path data/labels/cv_labels.json --lr 1e-4 --name multi_accent_finetune_25shot_5updates_philippines_joint_10_3_3_enc2_dec4_512_b6_22050hz --save-folder save/ --feat_extractor vgg_cnn --dropout 0.1 --num-enc-layers 2 --num-dec-layers 4 --num-heads 8 --dim-model 512 --dim-key 64 --dim-value 64 --dim-input 5120 --dim-inner 512 --dim-emb 512 --early-stop cer,50 --src-max-len 5000 --tgt-max-len 2500 --epochs 5 --sample-rate 22050 --continue-from save/joint_10_3_3_enc2_dec4_512_b6_22050hz/epoch_220000.th --beam-search --beam-width 5 --save-every 5 --opt_name sgd --evaluate-every 5 --training-mode joint &

###
# FINETUNE JOINT 50%
###
CUDA_VISIBLE_DEVICES=2 python3 finetune.py \
--train-manifest-list ./data/manifests/cv_20190612_philippines_train.csv \
--valid-manifest-list ./data/manifests/cv_20190612_philippines_test.csv \
--test-manifest-list ./data/manifests/cv_20190612_philippines_test.csv \
--train-partition-list 0.5 \
--cuda --k-train 6 --labels-path data/labels/cv_labels.json --lr 1e-4 --name multi_accent_finetune_50shot_5updates_philippines_joint_10_3_3_enc2_dec4_512_b6_22050hz --save-folder save/ --feat_extractor vgg_cnn --dropout 0.1 --num-enc-layers 2 --num-dec-layers 4 --num-heads 8 --dim-model 512 --dim-key 64 --dim-value 64 --dim-input 5120 --dim-inner 512 --dim-emb 512 --early-stop cer,50 --src-max-len 5000 --tgt-max-len 2500 --epochs 5 --sample-rate 22050 --continue-from save/joint_10_3_3_enc2_dec4_512_b6_22050hz/epoch_220000.th --beam-search --beam-width 5 --save-every 5 --opt_name sgd --evaluate-every 5 --training-mode joint &

###
# FINETUNE JOINT 100%
###
CUDA_VISIBLE_DEVICES=0 python3 finetune.py \
--train-manifest-list ./data/manifests/cv_20190612_philippines_train.csv \
--valid-manifest-list ./data/manifests/cv_20190612_philippines_test.csv \
--test-manifest-list ./data/manifests/cv_20190612_philippines_test.csv \
--train-partition-list 1 \
--cuda --k-train 6 --labels-path data/labels/cv_labels.json --lr 1e-4 --name multi_accent_finetune_100shot_5updates_philippines_joint_10_3_3_enc2_dec4_512_b6_22050hz --save-folder save/ --feat_extractor vgg_cnn --dropout 0.1 --num-enc-layers 2 --num-dec-layers 4 --num-heads 8 --dim-model 512 --dim-key 64 --dim-value 64 --dim-input 5120 --dim-inner 512 --dim-emb 512 --early-stop cer,50 --src-max-len 5000 --tgt-max-len 2500 --epochs 5 --sample-rate 22050 --continue-from save/joint_10_3_3_enc2_dec4_512_b6_22050hz/epoch_220000.th --beam-search --beam-width 5 --save-every 5 --opt_name sgd --evaluate-every 5 --training-mode joint &

##############################
# WALES
##############################

###
# FINETUNE META 10%
###
CUDA_VISIBLE_DEVICES=0 python3 finetune.py \
--train-manifest-list ./data/manifests/cv_20190612_wales_train.csv \
--valid-manifest-list ./data/manifests/cv_20190612_wales_test.csv \
--test-manifest-list ./data/manifests/cv_20190612_wales_test.csv \
--train-partition-list 0.1 \
--cuda --k-train 6 --labels-path data/labels/cv_labels.json --lr 1e-4 --name multi_accent_finetune_10shot_5updates_wales_maml_10_3_3_enc2_dec4_512_b6_22050hz_copy_grad_early10000 --save-folder save/ --feat_extractor vgg_cnn --dropout 0.1 --num-enc-layers 2 --num-dec-layers 4 --num-heads 8 --dim-model 512 --dim-key 64 --dim-value 64 --dim-input 5120 --dim-inner 512 --dim-emb 512 --early-stop cer,50 --src-max-len 5000 --tgt-max-len 2500 --epochs 5 --sample-rate 22050 --continue-from save/maml_10_3_3_enc2_dec4_512_b6_22050hz_copy_grad_early10000/epoch_220000.th --beam-search --beam-width 5 --save-every 5 --opt_name sgd --evaluate-every 5 &

###
# FINETUNE META 25%
###
CUDA_VISIBLE_DEVICES=0 python3 finetune.py \
--train-manifest-list ./data/manifests/cv_20190612_wales_train.csv \
--valid-manifest-list ./data/manifests/cv_20190612_wales_test.csv \
--test-manifest-list ./data/manifests/cv_20190612_wales_test.csv \
--train-partition-list 0.25 \
--cuda --k-train 6 --labels-path data/labels/cv_labels.json --lr 1e-4 --name multi_accent_finetune_25shot_5updates_wales_maml_10_3_3_enc2_dec4_512_b6_22050hz_copy_grad_early10000 --save-folder save/ --feat_extractor vgg_cnn --dropout 0.1 --num-enc-layers 2 --num-dec-layers 4 --num-heads 8 --dim-model 512 --dim-key 64 --dim-value 64 --dim-input 5120 --dim-inner 512 --dim-emb 512 --early-stop cer,50 --src-max-len 5000 --tgt-max-len 2500 --epochs 5 --sample-rate 22050 --continue-from save/maml_10_3_3_enc2_dec4_512_b6_22050hz_copy_grad_early10000/epoch_220000.th --beam-search --beam-width 5 --save-every 5 --opt_name sgd --evaluate-every 5 &

###
# FINETUNE META 50%
###
CUDA_VISIBLE_DEVICES=0 python3 finetune.py \
--train-manifest-list ./data/manifests/cv_20190612_wales_train.csv \
--valid-manifest-list ./data/manifests/cv_20190612_wales_test.csv \
--test-manifest-list ./data/manifests/cv_20190612_wales_test.csv \
--train-partition-list 0.5 \
--cuda --k-train 6 --labels-path data/labels/cv_labels.json --lr 1e-4 --name multi_accent_finetune_50shot_5updates_wales_maml_10_3_3_enc2_dec4_512_b6_22050hz_copy_grad_early10000 --save-folder save/ --feat_extractor vgg_cnn --dropout 0.1 --num-enc-layers 2 --num-dec-layers 4 --num-heads 8 --dim-model 512 --dim-key 64 --dim-value 64 --dim-input 5120 --dim-inner 512 --dim-emb 512 --early-stop cer,50 --src-max-len 5000 --tgt-max-len 2500 --epochs 5 --sample-rate 22050 --continue-from save/maml_10_3_3_enc2_dec4_512_b6_22050hz_copy_grad_early10000/epoch_220000.th --beam-search --beam-width 5 --save-every 5 --opt_name sgd --evaluate-every 5 &

###
# FINETUNE META 100%
###
CUDA_VISIBLE_DEVICES=2 python3 finetune.py \
--train-manifest-list ./data/manifests/cv_20190612_wales_train.csv \
--valid-manifest-list ./data/manifests/cv_20190612_wales_test.csv \
--test-manifest-list ./data/manifests/cv_20190612_wales_test.csv \
--train-partition-list 1 \
--cuda --k-train 6 --labels-path data/labels/cv_labels.json --lr 1e-4 --name multi_accent_finetune_100shot_5updates_wales_maml_10_3_3_enc2_dec4_512_b6_22050hz_copy_grad_early10000 --save-folder save/ --feat_extractor vgg_cnn --dropout 0.1 --num-enc-layers 2 --num-dec-layers 4 --num-heads 8 --dim-model 512 --dim-key 64 --dim-value 64 --dim-input 5120 --dim-inner 512 --dim-emb 512 --early-stop cer,50 --src-max-len 5000 --tgt-max-len 2500 --epochs 5 --sample-rate 22050 --continue-from save/maml_10_3_3_enc2_dec4_512_b6_22050hz_copy_grad_early10000/epoch_220000.th --beam-search --beam-width 5 --save-every 5 --opt_name sgd --evaluate-every 5 &

###
# FINETUNE JOINT 10%
###
CUDA_VISIBLE_DEVICES=0 python3 finetune.py \
--train-manifest-list ./data/manifests/cv_20190612_wales_train.csv \
--valid-manifest-list ./data/manifests/cv_20190612_wales_test.csv \
--test-manifest-list ./data/manifests/cv_20190612_wales_test.csv \
--train-partition-list 0.1 \
--cuda --k-train 6 --labels-path data/labels/cv_labels.json --lr 1e-4 --name multi_accent_finetune_10shot_5updates_wales_joint_10_3_3_enc2_dec4_512_b6_22050hz --save-folder save/ --feat_extractor vgg_cnn --dropout 0.1 --num-enc-layers 2 --num-dec-layers 4 --num-heads 8 --dim-model 512 --dim-key 64 --dim-value 64 --dim-input 5120 --dim-inner 512 --dim-emb 512 --early-stop cer,50 --src-max-len 5000 --tgt-max-len 2500 --epochs 5 --sample-rate 22050 --continue-from save/joint_10_3_3_enc2_dec4_512_b6_22050hz/epoch_220000.th --beam-search --beam-width 5 --save-every 5 --opt_name sgd --evaluate-every 5 --training-mode joint &

###
# FINETUNE JOINT 25%
###
CUDA_VISIBLE_DEVICES=1 python3 finetune.py \
--train-manifest-list ./data/manifests/cv_20190612_wales_train.csv \
--valid-manifest-list ./data/manifests/cv_20190612_wales_test.csv \
--test-manifest-list ./data/manifests/cv_20190612_wales_test.csv \
--train-partition-list 0.25 \
--cuda --k-train 6 --labels-path data/labels/cv_labels.json --lr 1e-4 --name multi_accent_finetune_25shot_5updates_wales_joint_10_3_3_enc2_dec4_512_b6_22050hz --save-folder save/ --feat_extractor vgg_cnn --dropout 0.1 --num-enc-layers 2 --num-dec-layers 4 --num-heads 8 --dim-model 512 --dim-key 64 --dim-value 64 --dim-input 5120 --dim-inner 512 --dim-emb 512 --early-stop cer,50 --src-max-len 5000 --tgt-max-len 2500 --epochs 5 --sample-rate 22050 --continue-from save/joint_10_3_3_enc2_dec4_512_b6_22050hz/epoch_220000.th --beam-search --beam-width 5 --save-every 5 --opt_name sgd --evaluate-every 5 --training-mode joint &

###
# FINETUNE JOINT 50%
###
CUDA_VISIBLE_DEVICES=0 python3 finetune.py \
--train-manifest-list ./data/manifests/cv_20190612_wales_train.csv \
--valid-manifest-list ./data/manifests/cv_20190612_wales_test.csv \
--test-manifest-list ./data/manifests/cv_20190612_wales_test.csv \
--train-partition-list 0.5 \
--cuda --k-train 6 --labels-path data/labels/cv_labels.json --lr 1e-4 --name multi_accent_finetune_50shot_5updates_wales_joint_10_3_3_enc2_dec4_512_b6_22050hz --save-folder save/ --feat_extractor vgg_cnn --dropout 0.1 --num-enc-layers 2 --num-dec-layers 4 --num-heads 8 --dim-model 512 --dim-key 64 --dim-value 64 --dim-input 5120 --dim-inner 512 --dim-emb 512 --early-stop cer,50 --src-max-len 5000 --tgt-max-len 2500 --epochs 5 --sample-rate 22050 --continue-from save/joint_10_3_3_enc2_dec4_512_b6_22050hz/epoch_220000.th --beam-search --beam-width 5 --save-every 5 --opt_name sgd --evaluate-every 5 --training-mode joint &

###
# FINETUNE JOINT 100%
###
CUDA_VISIBLE_DEVICES=1 python3 finetune.py \
--train-manifest-list ./data/manifests/cv_20190612_wales_train.csv \
--valid-manifest-list ./data/manifests/cv_20190612_wales_test.csv \
--test-manifest-list ./data/manifests/cv_20190612_wales_test.csv \
--train-partition-list 1 \
--cuda --k-train 6 --labels-path data/labels/cv_labels.json --lr 1e-4 --name multi_accent_finetune_100shot_5updates_wales_joint_10_3_3_enc2_dec4_512_b6_22050hz --save-folder save/ --feat_extractor vgg_cnn --dropout 0.1 --num-enc-layers 2 --num-dec-layers 4 --num-heads 8 --dim-model 512 --dim-key 64 --dim-value 64 --dim-input 5120 --dim-inner 512 --dim-emb 512 --early-stop cer,50 --src-max-len 5000 --tgt-max-len 2500 --epochs 5 --sample-rate 22050 --continue-from save/joint_10_3_3_enc2_dec4_512_b6_22050hz/epoch_220000.th --beam-search --beam-width 5 --save-every 5 --opt_name sgd --evaluate-every 5 --training-mode joint &

##############################
# BERMUDA
##############################

###
# FINETUNE META 10%
###
CUDA_VISIBLE_DEVICES=2 python3 finetune.py \
--train-manifest-list ./data/manifests/cv_20190612_bermuda_train.csv \
--valid-manifest-list ./data/manifests/cv_20190612_bermuda_test.csv \
--test-manifest-list ./data/manifests/cv_20190612_bermuda_test.csv \
--train-partition-list 0.1 \
--cuda --k-train 6 --labels-path data/labels/cv_labels.json --lr 1e-4 --name multi_accent_finetune_10shot_5updates_bermuda_maml_10_3_3_enc2_dec4_512_b6_22050hz_copy_grad_early10000 --save-folder save/ --feat_extractor vgg_cnn --dropout 0.1 --num-enc-layers 2 --num-dec-layers 4 --num-heads 8 --dim-model 512 --dim-key 64 --dim-value 64 --dim-input 5120 --dim-inner 512 --dim-emb 512 --early-stop cer,50 --src-max-len 5000 --tgt-max-len 2500 --epochs 5 --sample-rate 22050 --continue-from save/maml_10_3_3_enc2_dec4_512_b6_22050hz_copy_grad_early10000/epoch_220000.th --beam-search --beam-width 5 --save-every 5 --opt_name sgd --evaluate-every 5 &

###
# FINETUNE META 25%
###
CUDA_VISIBLE_DEVICES=1 python3 finetune.py \
--train-manifest-list ./data/manifests/cv_20190612_bermuda_train.csv \
--valid-manifest-list ./data/manifests/cv_20190612_bermuda_test.csv \
--test-manifest-list ./data/manifests/cv_20190612_bermuda_test.csv \
--train-partition-list 0.25 \
--cuda --k-train 6 --labels-path data/labels/cv_labels.json --lr 1e-4 --name multi_accent_finetune_25shot_5updates_bermuda_maml_10_3_3_enc2_dec4_512_b6_22050hz_copy_grad_early10000 --save-folder save/ --feat_extractor vgg_cnn --dropout 0.1 --num-enc-layers 2 --num-dec-layers 4 --num-heads 8 --dim-model 512 --dim-key 64 --dim-value 64 --dim-input 5120 --dim-inner 512 --dim-emb 512 --early-stop cer,50 --src-max-len 5000 --tgt-max-len 2500 --epochs 5 --sample-rate 22050 --continue-from save/maml_10_3_3_enc2_dec4_512_b6_22050hz_copy_grad_early10000/epoch_220000.th --beam-search --beam-width 5 --save-every 5 --opt_name sgd --evaluate-every 5 &

###
# FINETUNE META 50%
###
CUDA_VISIBLE_DEVICES=0 python3 finetune.py \
--train-manifest-list ./data/manifests/cv_20190612_bermuda_train.csv \
--valid-manifest-list ./data/manifests/cv_20190612_bermuda_test.csv \
--test-manifest-list ./data/manifests/cv_20190612_bermuda_test.csv \
--train-partition-list 0.5 \
--cuda --k-train 6 --labels-path data/labels/cv_labels.json --lr 1e-4 --name multi_accent_finetune_50shot_5updates_bermuda_maml_10_3_3_enc2_dec4_512_b6_22050hz_copy_grad_early10000 --save-folder save/ --feat_extractor vgg_cnn --dropout 0.1 --num-enc-layers 2 --num-dec-layers 4 --num-heads 8 --dim-model 512 --dim-key 64 --dim-value 64 --dim-input 5120 --dim-inner 512 --dim-emb 512 --early-stop cer,50 --src-max-len 5000 --tgt-max-len 2500 --epochs 5 --sample-rate 22050 --continue-from save/maml_10_3_3_enc2_dec4_512_b6_22050hz_copy_grad_early10000/epoch_220000.th --beam-search --beam-width 5 --save-every 5 --opt_name sgd --evaluate-every 5 &

###
# FINETUNE META 100%
###
CUDA_VISIBLE_DEVICES=2 python3 finetune.py \
--train-manifest-list ./data/manifests/cv_20190612_bermuda_train.csv \
--valid-manifest-list ./data/manifests/cv_20190612_bermuda_test.csv \
--test-manifest-list ./data/manifests/cv_20190612_bermuda_test.csv \
--train-partition-list 1 \
--cuda --k-train 6 --labels-path data/labels/cv_labels.json --lr 1e-4 --name multi_accent_finetune_100shot_5updates_bermuda_maml_10_3_3_enc2_dec4_512_b6_22050hz_copy_grad_early10000 --save-folder save/ --feat_extractor vgg_cnn --dropout 0.1 --num-enc-layers 2 --num-dec-layers 4 --num-heads 8 --dim-model 512 --dim-key 64 --dim-value 64 --dim-input 5120 --dim-inner 512 --dim-emb 512 --early-stop cer,50 --src-max-len 5000 --tgt-max-len 2500 --epochs 5 --sample-rate 22050 --continue-from save/maml_10_3_3_enc2_dec4_512_b6_22050hz_copy_grad_early10000/epoch_220000.th --beam-search --beam-width 5 --save-every 5 --opt_name sgd --evaluate-every 5 &

###
# FINETUNE JOINT 10%
###
CUDA_VISIBLE_DEVICES=2 python3 finetune.py \
--train-manifest-list ./data/manifests/cv_20190612_bermuda_train.csv \
--valid-manifest-list ./data/manifests/cv_20190612_bermuda_test.csv \
--test-manifest-list ./data/manifests/cv_20190612_bermuda_test.csv \
--train-partition-list 0.1 \
--cuda --k-train 6 --labels-path data/labels/cv_labels.json --lr 1e-4 --name multi_accent_finetune_10shot_5updates_bermuda_joint_10_3_3_enc2_dec4_512_b6_22050hz --save-folder save/ --feat_extractor vgg_cnn --dropout 0.1 --num-enc-layers 2 --num-dec-layers 4 --num-heads 8 --dim-model 512 --dim-key 64 --dim-value 64 --dim-input 5120 --dim-inner 512 --dim-emb 512 --early-stop cer,50 --src-max-len 5000 --tgt-max-len 2500 --epochs 5 --sample-rate 22050 --continue-from save/joint_10_3_3_enc2_dec4_512_b6_22050hz/epoch_220000.th --beam-search --beam-width 5 --save-every 5 --opt_name sgd --evaluate-every 5 --training-mode joint &

###
# FINETUNE JOINT 25%
###
CUDA_VISIBLE_DEVICES=1 python3 finetune.py \
--train-manifest-list ./data/manifests/cv_20190612_bermuda_train.csv \
--valid-manifest-list ./data/manifests/cv_20190612_bermuda_test.csv \
--test-manifest-list ./data/manifests/cv_20190612_bermuda_test.csv \
--train-partition-list 0.25 \
--cuda --k-train 6 --labels-path data/labels/cv_labels.json --lr 1e-4 --name multi_accent_finetune_25shot_5updates_bermuda_joint_10_3_3_enc2_dec4_512_b6_22050hz --save-folder save/ --feat_extractor vgg_cnn --dropout 0.1 --num-enc-layers 2 --num-dec-layers 4 --num-heads 8 --dim-model 512 --dim-key 64 --dim-value 64 --dim-input 5120 --dim-inner 512 --dim-emb 512 --early-stop cer,50 --src-max-len 5000 --tgt-max-len 2500 --epochs 5 --sample-rate 22050 --continue-from save/joint_10_3_3_enc2_dec4_512_b6_22050hz/epoch_220000.th --beam-search --beam-width 5 --save-every 5 --opt_name sgd --evaluate-every 5 --training-mode joint &

###
# FINETUNE JOINT 50%
###
CUDA_VISIBLE_DEVICES=2 python3 finetune.py \
--train-manifest-list ./data/manifests/cv_20190612_bermuda_train.csv \
--valid-manifest-list ./data/manifests/cv_20190612_bermuda_test.csv \
--test-manifest-list ./data/manifests/cv_20190612_bermuda_test.csv \
--train-partition-list 0.5 \
--cuda --k-train 6 --labels-path data/labels/cv_labels.json --lr 1e-4 --name multi_accent_finetune_50shot_5updates_bermuda_joint_10_3_3_enc2_dec4_512_b6_22050hz --save-folder save/ --feat_extractor vgg_cnn --dropout 0.1 --num-enc-layers 2 --num-dec-layers 4 --num-heads 8 --dim-model 512 --dim-key 64 --dim-value 64 --dim-input 5120 --dim-inner 512 --dim-emb 512 --early-stop cer,50 --src-max-len 5000 --tgt-max-len 2500 --epochs 5 --sample-rate 22050 --continue-from save/joint_10_3_3_enc2_dec4_512_b6_22050hz/epoch_220000.th --beam-search --beam-width 5 --save-every 5 --opt_name sgd --evaluate-every 5 --training-mode joint &

###
# FINETUNE JOINT 100%
###
CUDA_VISIBLE_DEVICES=0 python3 finetune.py \
--train-manifest-list ./data/manifests/cv_20190612_bermuda_train.csv \
--valid-manifest-list ./data/manifests/cv_20190612_bermuda_test.csv \
--test-manifest-list ./data/manifests/cv_20190612_bermuda_test.csv \
--train-partition-list 1 \
--cuda --k-train 6 --labels-path data/labels/cv_labels.json --lr 1e-4 --name multi_accent_finetune_100shot_5updates_bermuda_joint_10_3_3_enc2_dec4_512_b6_22050hz --save-folder save/ --feat_extractor vgg_cnn --dropout 0.1 --num-enc-layers 2 --num-dec-layers 4 --num-heads 8 --dim-model 512 --dim-key 64 --dim-value 64 --dim-input 5120 --dim-inner 512 --dim-emb 512 --early-stop cer,50 --src-max-len 5000 --tgt-max-len 2500 --epochs 5 --sample-rate 22050 --continue-from save/joint_10_3_3_enc2_dec4_512_b6_22050hz/epoch_220000.th --beam-search --beam-width 5 --save-every 5 --opt_name sgd --evaluate-every 5 --training-mode joint &