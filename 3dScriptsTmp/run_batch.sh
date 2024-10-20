#!/bin/bash
#conda activate lightgaussian
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtiff.so.5
# /data2/lpl/data/pure

#查询可用GPU
steps=("sfm" "3dgs")

#scene_list=("bottle_f00" "gourd_fA" "perfume_f05" "smallPart1_fA" "smallPart2_fA" "wrench_fA" "toy_fA")
#type_list=("so_images" "so_images" "so_images" "so_images" "so_images" "so_images" "so_images")
#status_list=(0 0 0 0 0 0 0)
#scene_list=("shelf_f04" "smallPart2_f04M02" "smallPart2_f04M02_CP" "toy_fA_CP")
#type_list=("so_images_2" "so_images_2" "so_images_2c" "so_images")
#status_list=(0 0 0 0)

#scene_list=("smallPart2_f04M02_CP1" "smallPart_f02" "toy_fA_CP")
#type_list=("so_images_2c" "so_images_2" "so_images")
#status_list=(0 0 0)

scene_list=("toy_fA_CP2")
type_list=("so_images")
status_list=(0)

input_path="/app/input"
out_path="/app/input"
output_file="batch_train_status.txt"
> "$output_file"
# 生成 1000 到 9999 之间的随机整数
#seed=23456
#RANDOM=$seed  # 固定随机种子


is_element_in_list() {
    local element="$1"
    local steps=("${@:2}")  # 从第二个参数开始获取列表

    for item in "${steps[@]}"; do
        if [ "$item" == "$element" ]; then
            return 0  # 返回0表示找到元素
        fi
    done
    return 1  # 返回1表示未找到元素
}

get_available_gpu() {
    local mem_threshold=5000
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -v threshold="$mem_threshold" -F ',' '$2 < threshold { print $1; exit }'
}

# 获取两个列表的长度
listLength=${#scene_list[@]}

for ((i=0; i<listLength; i++)); do
    # 检查状态
    if [ ${status_list[i]} -ne 1 ]; then
        gpu_id=$(get_available_gpu)
        while  [[ -z $gpu_id ]]; do
            echo "No GPU available at the moment. Retrying in 1 minute."
            sleep 60
            gpu_id=$(get_available_gpu)
        done

        scene=${scene_list[i]}
        type=${type_list[i]}
        if [ ! -d "${out_path}/${scene}" ];then
            mkdir ${out_path}/${scene}
        else
            echo "文件夹${out_path}/${scene}已经存在"
        fi

        start_time=$(date +%s)
        element="sfm"
        if is_element_in_list "$element" "${steps[@]}"; then
            python sfm.py --input_path ${input_path}/${scene}  --out_path ${out_path}/${scene} --type ${type}
            echo "$element command success"
        fi
        sfm_time=$(date +%s)
        element="3dgs"
        if is_element_in_list "$element" "${steps[@]}"; then
            random_number=$((RANDOM % 9000 + 1000))
            CUDA_VISIBLE_DEVICES=$gpu_id python 3dgs.py --input_path ${input_path}/${scene}  --out_path ${out_path}/${scene} --type ${type} --port "$random_number"
            echo "$element command success"
        fi
        gs_time=$(date +%s)

        sfm_elapsed_time=$((sfm_time-start_time))
        gs_elapsed_time=$((gs_time - sfm_time))
        # 转换为分钟和秒
        sfm_hours=$((sfm_elapsed_time / 3600))
        sfm_minutes=$(((sfm_elapsed_time % 3600) / 60))
        sfm_seconds=$((sfm_elapsed_time % 60))
        gs_hours=$((gs_elapsed_time / 3600))
        gs_minutes=$(((gs_elapsed_time % 3600) / 60))
        gs_seconds=$((gs_elapsed_time % 60))
        echo "===============================================" >> "$output_file"
        echo "场景名: ${scene}" >> "$output_file"
        echo "sfm总用时: ${sfm_hours}:${sfm_minutes}:${sfm_seconds}" >> "$output_file"
        echo "高斯总用时: ${gs_hours}:${gs_minutes}:${gs_seconds}" >> "$output_file"
        echo "===============================================" >> "$output_file"

        status_list[i]=1
    fi

done

