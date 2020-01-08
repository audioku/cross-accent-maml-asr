CUDA_VISIBLE_DEVICES=1 python3 joint_train.py \
--train-manifest-list ./data/manifests/cv_20190612_philippines_train.csv \
--valid-manifest-list ./data/manifests/cv_20190612_philippines_test.csv \
--test-manifest-list ./data/manifests/cv_20190612_philippines_test.csv \
--cuda --k-train 6 --labels-path data/labels/cv_labels.json --lr 1e-4 --name philippines_enc2_dec4_512_b6_22050hz --save-folder save/ --save-every 10000 --feat_extractor vgg_cnn --dropout 0.1 --num-enc-layers 2 --num-dec-layers 4 --num-heads 8 --dim-model 512 --dim-key 64 --dim-value 64 --dim-input 5120 --dim-inner 512 --dim-emb 512 --early-stop cer,20 --src-max-len 5000 --tgt-max-len 2500 --evaluate-every 1000 --epochs 500000 --sample-rate 22050 --train-partition-list 1

CUDA_VISIBLE_DEVICES=2 python3 joint_train.py \
--train-manifest-list ./data/manifests/cv_20190612_wales_train.csv \
--valid-manifest-list ./data/manifests/cv_20190612_wales_test.csv \
--test-manifest-list ./data/manifests/cv_20190612_wales_test.csv \
--cuda --k-train 6 --labels-path data/labels/cv_labels.json --lr 1e-4 --name wales_enc2_dec4_512_b6_22050hz --save-folder save/ --save-every 10000 --feat_extractor vgg_cnn --dropout 0.1 --num-enc-layers 2 --num-dec-layers 4 --num-heads 8 --dim-model 512 --dim-key 64 --dim-value 64 --dim-input 5120 --dim-inner 512 --dim-emb 512 --early-stop cer,20 --src-max-len 5000 --tgt-max-len 2500 --evaluate-every 1000 --epochs 500000 --sample-rate 22050 --train-partition-list 1

CUDA_VISIBLE_DEVICES=0 python3 joint_train.py \
--train-manifest-list ./data/manifests/cv_20190612_bermuda_train.csv \
--valid-manifest-list ./data/manifests/cv_20190612_bermuda_test.csv \
--test-manifest-list ./data/manifests/cv_20190612_bermuda_test.csv \
--cuda --k-train 6 --labels-path data/labels/cv_labels.json --lr 1e-4 --name bermuda_enc2_dec4_512_b6_22050hz --save-folder save/ --save-every 10000 --feat_extractor vgg_cnn --dropout 0.1 --num-enc-layers 2 --num-dec-layers 4 --num-heads 8 --dim-model 512 --dim-key 64 --dim-value 64 --dim-input 5120 --dim-inner 512 --dim-emb 512 --early-stop cer,20 --src-max-len 5000 --tgt-max-len 2500 --evaluate-every 1000 --epochs 500000 --sample-rate 22050 --train-partition-list 1