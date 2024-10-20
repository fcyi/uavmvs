#!/bin/bash
#conda activate lightgaussian
# /data2/lpl/data/pure

steps=("sfmDepth")

scene_list=("5")
type_list=("glomap_eval")
time_list=("SBD")

status_list=(0)

input_path="/app/input/nonTrain/hqd_bottle"
out_path="/app/input/nonTrain/hqd_bottle"
output_file="batch_train_status.txt"

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
        echo "========================= start processing =========================="

        scene=${scene_list[i]}
        type=${type_list[i]}
        timeT=${time_list[i]}

        inputPath="${input_path}/${scene}"
        outputPath="${out_path}/${scene}_${timeT}/"

#        if [ ! -d "${outputPath}" ];then
#            mkdir ${outputPath}
#        else
#            echo "文件夹${outputPath}已经存在"
#        fi
#
#        # 测试标准数据集需要增加这部分拷贝内容
#        if [ ! -d "${outputPath}/train" ]; then
#            # 如果文件夹不存在，拷贝文件和文件夹
#            cp -r "${inputPath}/." "${outputPath}/"
#            echo "所有测试输入数据已拷贝到${outputPath}/"
#        else
#            echo "测试数据已存在，不需要拷贝。"
#        fi

        start_time=$(date +%s)

        rmpv_time=$(date +%s)

        element="sfmDepth"
        if is_element_in_list "$element" "${steps[@]}"; then
            python sparse_based_fixCamParamsAndDepth.py --createDir ${outputPath} --useGivenIntrinsicParams true --fov 90 --isInv true --camModel "SIMPLE_PINHOLE" --BOWPath "/app/input/3drecon/src/sfm/resources/vocab_tree_flickr100K_words256K.bin" --filterType 1 --zLimits "1000,4,40,200,1600000,1700000"
            echo "$element command success"
        fi

        element="sfmTran"
        if is_element_in_list "$element" "${steps[@]}"; then
            python sparse_based_fixCamParams.py --createDir ${outputPath} --useGivenIntrinsicParams ture --fov 90 --isInv true --camModel "SIMPLE_PINHOLE" --BOWPath "/app/input/3drecon/src/sfm/resources/vocab_tree_flickr100K_words256K.bin"
            echo "$element command success"
        fi

        sfm_time=$(date +%s)

        rmpv_elapsed_time=$((rmpv_time-start_time))
        sfm_elapsed_time=$((sfm_time-rmpv_time))
        # 转换为分钟和秒
        sfm_hours=$((sfm_elapsed_time / 3600))
        sfm_minutes=$(((sfm_elapsed_time % 3600) / 60))
        sfm_seconds=$((sfm_elapsed_time % 60))

        outputFilePath="${outputPath}/${output_file}"
        echo "===============================================" >> "$outputFilePath"
        echo $(date "+%Y_%m_%d_%H_%M")
        stepLen=${#steps[@]}
        stepStr=""
        for ((stepj=0; stepj<stepLen; stepj++)); do
           stepStr="${stepStr}_${steps[stepj]}"
        done
        echo "steps: ${stepStr}" >> "$outputFilePath"
        echo "场景名: ${scene}" >> "$outputFilePath"
        echo "sfm总用时: ${sfm_hours}:${sfm_minutes}:${sfm_seconds}" >> "$outputFilePath"
        echo "===============================================" >> "$outputFilePath"
        status_list[i]=1
    fi

done
