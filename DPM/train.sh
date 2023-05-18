#!/bin/bash
for dataset in 1; do
    for lrs in 0.00015; do
        for (( i=0; i <= 1; i++ )); do
            for num in 0 1 2 3; do
                seedtmp=$((num + 1))
                seedm=$((i * 4))
                seedfor=$((seedm + seedtmp))
                #CUDA_VISIBLE_DEVICES=1 python train_nri_warmup_result.py --start_steps 20000 --n_episode 50 --lr $lrs --num_steps 18000 --rolling_steps 50 --nri_d 32 --seed $((i + num)) --stocks 0 --smoothing_days 5 --cnn_d 50 --nri_shuffle 20 --nri_lr 0.00015 --cnn_d2 25 --L2_w 0 --L3_w 0 &
                #specific dataset

                CUDA_VISIBLE_DEVICES=$num python main.py  --model_name dpm_v2 --n_episode 50 --lr $lrs --num_steps 18000 --rolling_steps 50 --nri_d 32 --seed $seedfor --stocks $dataset --smoothing_days 5 --cnn_d 50 --nri_shuffle 20 --nri_lr 0.00015 --cnn_d2 25 --L2_w 0 --L3_w 0 & 



            done
            wait
        done
    done
done

#for dataset in 0 --L3_w 0; do
#    for lrs in 0.00012; do
#        for (( i=0; i <= 1; i++ )); do
#            for num in 0 1 2 3; do
#                seedtmp=$((num + 1))
#               seedm=$((i * 4))
#                seedfor=$((seedm + seedtmp))
#                #CUDA_VISIBLE_DEVICES=1 python train_nri_warmup_result.py --start_steps 20000 --n_episode 50 --lr $lrs --num_steps 18000 --rolling_steps 50 --nri_d 32 --seed $((i + num)) --stocks 0 --smoothing_days 5 --cnn_d 50 --nri_shuffle 20 --nri_lr 0.00015 --cnn_d2 25 --L2_w 0 --L3_w 0 &
#                #specific dataset
#                CUDA_VISIBLE_DEVICES=$num python  train_nri_no_val.py  --n_episode 20 --lr $lrs --num_steps 18000 --rolling_steps 30 --nri_d 32 --seed $seedfor --stocks $dataset --smoothing_days 5 --cnn_d 20 --nri_shuffle 10 --nri_lr 0.00012 --cnn_d2 10 --L3_w 1.5 & 
#
#            done
#            wait
#        done
#    done
#done

#for s_day in 5; do
#    for lrs in 0.00015; do
#        for (( i=3; i <= 3; i++ )); do
#            for num in 0; do
#                seedtmp=$((num + 1))
#                seedm=$((i * 1))
#                seedfor=$((seedm + seedtmp))
#                #other dataset
#                CUDA_VISIBLE_DEVICES=1 python train_nri_no_val.py  --n_episode 50 --lr $lrs --num_steps 18000 --rolling_steps 50 --nri_d 32 --seed $seedfor --stocks 3 --smoothing_days $s_day --cnn_d 50 --nri_shuffle 20 --nri_lr 0.00015 --cnn_d2 25 --L3_w 0&
#            done
#            wait
#        done
#
#    done
#done
