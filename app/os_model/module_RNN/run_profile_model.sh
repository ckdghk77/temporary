available_gpus=(0 1 2 3 4 5 6 7)

temp_file=tmp_gpu_dne
echo -n ${available_gpus[@]} >| $temp_file

seeds=(0 1 2 3 4)
var_tars=(primacy recency lr ablation)
models=(Transformer)
#models=(BiLSTM_Attn2)
dropouts=(0.1)


for seed in ${seeds[@]};
do

    for var_tar in ${var_tars[@]};
    do
        for model in ${models[@]};
        do
            for dropout in ${dropouts[@]};
            do
                #### start line for gpu scheduling
                line=$(head -1 $temp_file)
                available_gpus=(${line//^0-9})
                declare -a available_gpus

                while [ ${#available_gpus[@]} = 0 ]
                do
                    #echo "Waiting" {$available_gpus[@]}
                    sleep 1
                    line=$(head -1 $temp_file)
                    available_gpus=(${line//^0-9})
                    declare -a available_gpus
                done

                gpu_idx=${available_gpus[0]}
                available_gpus=(${available_gpus[@]/$gpu_idx})
                echo -n ${available_gpus[@]} >| $temp_file
                #### end of line for gpu scheduling

                { CUDA_VISIBLE_DEVICES=$gpu_idx python dne_profiling_model.py --seed=$seed --model=$model --var_tar=$var_tar \
                                                            --dropout=$dropout ; echo -n " "$gpu_idx >> $temp_file ; } &\

                sleep 1
            done
        done
    done
done
