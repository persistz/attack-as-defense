input_file_name_list=("benign"
                      "fgsm"
                      "bim-a"
                      "jsma"
                      "cw")

num=${#input_file_name_list[@]}
for id in `seq 0 $((num-1))`
    do
        wait
        CUDA_VISIBLE_DEVICES=3 python attack_iter_generate.py -a BIM -i ${input_file_name_list[$id]}
    done

wait
for id in `seq 0 $((num-1))`
    do
        wait
        CUDA_VISIBLE_DEVICES=3 python attack_iter_generate.py -a BIM2 -i ${input_file_name_list[$id]}
    done

wait
for id in `seq 0 $((num-1))`
    do
        wait
        CUDA_VISIBLE_DEVICES=3 python attack_iter_generate.py -a JSMA -i ${input_file_name_list[$id]}
    done