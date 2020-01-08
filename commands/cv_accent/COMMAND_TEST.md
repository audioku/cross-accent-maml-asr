CUDA_VISIBLE_DEVICES=0 python3 test.py \
--test-manifest-list ./data/manifests/cv_20190612_philippines_test.csv \
--cuda --labels-path data/labels/cv_labels.json --lr 1e-4 --training-mode meta --continue-from save/maml_10_3_3_enc2_dec4_512_b6_22050hz_copy_grad_early10000/epoch_220000.th --tgt-max-len 150 --k-test 1 --beam-search --beam-width 5

CUDA_VISIBLE_DEVICES=1 python3 test.py \
--test-manifest-list ./data/manifests/cv_20190612_philippines_test.csv \
--cuda --labels-path data/labels/cv_labels.json --lr 1e-4 --training-mode joint --continue-from save/joint_10_3_3_enc2_dec4_512_b6_22050hz/epoch_220000.th --tgt-max-len 150 --k-test 1 --beam-search --beam-width 5

CUDA_VISIBLE_DEVICES=2 python3 test.py \
--test-manifest-list ./data/manifests/cv_20190612_wales_test.csv \
--cuda --labels-path data/labels/cv_labels.json --lr 1e-4 --training-mode meta --continue-from save/maml_10_3_3_enc2_dec4_512_b6_22050hz_copy_grad_early10000/epoch_220000.th --tgt-max-len 150 --k-test 1 --beam-search --beam-width 5

CUDA_VISIBLE_DEVICES=2 python3 test.py \
--test-manifest-list ./data/manifests/cv_20190612_wales_test.csv \
--cuda --labels-path data/labels/cv_labels.json --lr 1e-4 --training-mode joint --continue-from save/joint_10_3_3_enc2_dec4_512_b6_22050hz/epoch_220000.th --tgt-max-len 150 --k-test 1 --beam-search --beam-width 5

CUDA_VISIBLE_DEVICES=0 python3 test.py \
--test-manifest-list ./data/manifests/cv_20190612_bermuda_test.csv \
--cuda --labels-path data/labels/cv_labels.json --lr 1e-4 --training-mode meta --continue-from save/maml_10_3_3_enc2_dec4_512_b6_22050hz_copy_grad_early10000/epoch_220000.th --tgt-max-len 150 --k-test 1 --beam-search --beam-width 5

CUDA_VISIBLE_DEVICES=1 python3 test.py \
--test-manifest-list ./data/manifests/cv_20190612_bermuda_test.csv \
--cuda --labels-path data/labels/cv_labels.json --lr 1e-4 --training-mode joint --continue-from save/joint_10_3_3_enc2_dec4_512_b6_22050hz/epoch_220000.th --tgt-max-len 150 --k-test 1 --beam-search --beam-width 5