###
# RUN META LEARNING 
###
CUDA_VISIBLE_DEVICES=0 python3 meta_train.py \
--train-manifest-list ./data/manifests/cv_20190612_us.csv ./data/manifests/cv_20190612_england.csv ./data/manifests/cv_20190612_indian.csv ./data/manifests/cv_20190612_australia.csv ./data/manifests/cv_20190612_newzealand.csv ./data/manifests/cv_20190612_african.csv  ./data/manifests/cv_20190612_ireland.csv ./data/manifests/cv_20190612_hongkong.csv ./data/manifests/cv_20190612_malaysia.csv ./data/manifests/cv_20190612_singapore.csv \
--valid-manifest-list ./data/manifests/cv_20190612_canada.csv ./data/manifests/cv_20190612_scotland.csv ./data/manifests/cv_20190612_southatlandtic.csv \
--test-manifest-list ./data/manifests/cv_20190612_philippines.csv ./data/manifests/cv_20190612_wales.csv ./data/manifests/cv_20190612_bermuda.csv \
--cuda --k-train 6 --k-valid 6 --labels-path data/labels/cv_labels.json --lr 1e-4 --name maml_10_3_3_enc2_dec4_512_b6_22050hz_copy_grad --save-folder save/ --save-every 10000 --feat_extractor vgg_cnn --dropout 0.1 --num-enc-layers 2 --num-dec-layers 4 --num-heads 8 --dim-model 512 --dim-key 64 --dim-value 64 --dim-input 5120 --dim-inner 512 --dim-emb 512 --early-stop cer,50 --src-max-len 5000 --tgt-max-len 2500 --evaluate-every 100 --epochs 500000 --sample-rate 22050 --copy-grad --num-meta-test 10

###
# RUN META LEARNING + Adversarial
###
CUDA_VISIBLE_DEVICES=5 python3 meta_train.py \
--train-manifest-list ./data/manifests/cv_20190612_us.csv ./data/manifests/cv_20190612_england.csv ./data/manifests/cv_20190612_indian.csv ./data/manifests/cv_20190612_australia.csv ./data/manifests/cv_20190612_newzealand.csv ./data/manifests/cv_20190612_african.csv  ./data/manifests/cv_20190612_ireland.csv ./data/manifests/cv_20190612_hongkong.csv ./data/manifests/cv_20190612_malaysia.csv ./data/manifests/cv_20190612_singapore.csv \
--valid-manifest-list ./data/manifests/cv_20190612_canada.csv ./data/manifests/cv_20190612_scotland.csv ./data/manifests/cv_20190612_southatlandtic.csv \
--test-manifest-list ./data/manifests/cv_20190612_philippines.csv ./data/manifests/cv_20190612_wales.csv ./data/manifests/cv_20190612_bermuda.csv \
--cuda --k-train 6 --k-valid 6 --labels-path data/labels/cv_labels.json --lr 1e-4 --name maml_adv_10_3_3_enc2_dec4_512_b6_22050hz_copy_grad --save-folder save/ --save-every 10000 --feat_extractor vgg_cnn --dropout 0.1 --num-enc-layers 2 --num-dec-layers 4 --num-heads 8 --dim-model 512 --dim-key 64 --dim-value 64 --dim-input 5120 --dim-inner 512 --dim-emb 512 --early-stop cer,50 --src-max-len 5000 --tgt-max-len 2500 --evaluate-every 100 --epochs 500000 --sample-rate 22050 --copy-grad --num-meta-test 10 --adversarial

###
# RUN JOINT LEARNING 
###
CUDA_VISIBLE_DEVICES=1 python3 joint_train.py \
--train-manifest-list ./data/manifests/cv_20190612_us.csv ./data/manifests/cv_20190612_england.csv ./data/manifests/cv_20190612_indian.csv ./data/manifests/cv_20190612_australia.csv ./data/manifests/cv_20190612_newzealand.csv ./data/manifests/cv_20190612_african.csv  ./data/manifests/cv_20190612_ireland.csv ./data/manifests/cv_20190612_hongkong.csv ./data/manifests/cv_20190612_malaysia.csv ./data/manifests/cv_20190612_singapore.csv \
--valid-manifest-list ./data/manifests/cv_20190612_canada.csv ./data/manifests/cv_20190612_scotland.csv ./data/manifests/cv_20190612_southatlandtic.csv \
--test-manifest-list ./data/manifests/cv_20190612_philippines.csv ./data/manifests/cv_20190612_wales.csv ./data/manifests/cv_20190612_bermuda.csv \
--cuda --k-train 6 --labels-path data/labels/cv_labels.json --lr 1e-4 --name joint_10_3_3_enc2_dec4_512_b6_22050hz --save-folder save/ --save-every 10000 --feat_extractor vgg_cnn --dropout 0.1 --num-enc-layers 2 --num-dec-layers 4 --num-heads 8 --dim-model 512 --dim-key 64 --dim-value 64 --dim-input 5120 --dim-inner 512 --dim-emb 512 --early-stop cer,20 --src-max-len 5000 --tgt-max-len 2500 --evaluate-every 1000 --epochs 500000 --sample-rate 22050 --train-partition-list 1 1 1 1 1 1 1 1 1 1

###
# RUN JOINT LEARNING + Adversarial
###
CUDA_VISIBLE_DEVICES=3 python3 joint_train.py \
--train-manifest-list ./data/manifests/cv_20190612_us.csv ./data/manifests/cv_20190612_england.csv ./data/manifests/cv_20190612_indian.csv ./data/manifests/cv_20190612_australia.csv ./data/manifests/cv_20190612_newzealand.csv ./data/manifests/cv_20190612_african.csv  ./data/manifests/cv_20190612_ireland.csv ./data/manifests/cv_20190612_hongkong.csv ./data/manifests/cv_20190612_malaysia.csv ./data/manifests/cv_20190612_singapore.csv \
--valid-manifest-list ./data/manifests/cv_20190612_canada.csv ./data/manifests/cv_20190612_scotland.csv ./data/manifests/cv_20190612_southatlandtic.csv \
--test-manifest-list ./data/manifests/cv_20190612_philippines.csv ./data/manifests/cv_20190612_wales.csv ./data/manifests/cv_20190612_bermuda.csv \
--cuda --k-train 6 --labels-path data/labels/cv_labels.json --lr 1e-4 --name joint_adv_10_3_3_enc2_dec4_512_b6_22050hz --save-folder save/ --save-every 10000 --feat_extractor vgg_cnn --dropout 0.1 --num-enc-layers 2 --num-dec-layers 4 --num-heads 8 --dim-model 512 --dim-key 64 --dim-value 64 --dim-input 5120 --dim-inner 512 --dim-emb 512 --early-stop cer,20 --src-max-len 5000 --tgt-max-len 2500 --evaluate-every 1000 --epochs 500000 --sample-rate 22050 --train-partition-list 1 1 1 1 1 1 1 1 1 1 --adversarial

###
# RUN JOINT LEARNING + Adversarial + Beta_decay
###
CUDA_VISIBLE_DEVICES=3 python3 joint_train.py \
--train-manifest-list ./data/manifests/cv_20190612_us.csv ./data/manifests/cv_20190612_england.csv ./data/manifests/cv_20190612_indian.csv ./data/manifests/cv_20190612_australia.csv ./data/manifests/cv_20190612_newzealand.csv ./data/manifests/cv_20190612_african.csv  ./data/manifests/cv_20190612_ireland.csv ./data/manifests/cv_20190612_hongkong.csv ./data/manifests/cv_20190612_malaysia.csv ./data/manifests/cv_20190612_singapore.csv \
--valid-manifest-list ./data/manifests/cv_20190612_canada.csv ./data/manifests/cv_20190612_scotland.csv ./data/manifests/cv_20190612_southatlandtic.csv \
--test-manifest-list ./data/manifests/cv_20190612_philippines.csv ./data/manifests/cv_20190612_wales.csv ./data/manifests/cv_20190612_bermuda.csv \
--cuda --k-train 6 --labels-path data/labels/cv_labels.json --lr 1e-4 --name joint_adv_decay_10_3_3_enc2_dec4_512_b6_22050hz --save-folder save/ --save-every 10000 --feat_extractor vgg_cnn --dropout 0.1 --num-enc-layers 2 --num-dec-layers 4 --num-heads 8 --dim-model 512 --dim-key 64 --dim-value 64 --dim-input 5120 --dim-inner 512 --dim-emb 512 --early-stop cer,20 --src-max-len 5000 --tgt-max-len 2500 --evaluate-every 1000 --epochs 500000 --sample-rate 22050 --train-partition-list 1 1 1 1 1 1 1 1 1 1 --adversarial --beta-decay

###
# RUN JOINT LEARNING + Multi-Task
###
CUDA_VISIBLE_DEVICES=3 python3 joint_train.py \
--train-manifest-list ./data/manifests/cv_20190612_us.csv ./data/manifests/cv_20190612_england.csv ./data/manifests/cv_20190612_indian.csv ./data/manifests/cv_20190612_australia.csv ./data/manifests/cv_20190612_newzealand.csv ./data/manifests/cv_20190612_african.csv  ./data/manifests/cv_20190612_ireland.csv ./data/manifests/cv_20190612_hongkong.csv ./data/manifests/cv_20190612_malaysia.csv ./data/manifests/cv_20190612_singapore.csv \
--valid-manifest-list ./data/manifests/cv_20190612_canada.csv ./data/manifests/cv_20190612_scotland.csv ./data/manifests/cv_20190612_southatlandtic.csv \
--test-manifest-list ./data/manifests/cv_20190612_philippines.csv ./data/manifests/cv_20190612_wales.csv ./data/manifests/cv_20190612_bermuda.csv \
--cuda --k-train 6 --labels-path data/labels/cv_labels.json --lr 1e-4 --name joint_multitask_decay_10_3_3_enc2_dec4_512_b6_22050hz --save-folder save/ --save-every 10000 --feat_extractor vgg_cnn --dropout 0.1 --num-enc-layers 2 --num-dec-layers 4 --num-heads 8 --dim-model 512 --dim-key 64 --dim-value 64 --dim-input 5120 --dim-inner 512 --dim-emb 512 --early-stop cer,20 --src-max-len 5000 --tgt-max-len 2500 --evaluate-every 1000 --epochs 500000 --sample-rate 22050 --train-partition-list 1 1 1 1 1 1 1 1 1 1 --multitask