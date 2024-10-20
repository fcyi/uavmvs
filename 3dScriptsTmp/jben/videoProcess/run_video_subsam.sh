#!/bin/bash

# # 检查是否提供了必要的参数
# if [ "$#" -ne 3 ]; then
#     echo "用法: $0 <输入视频文件> <抽帧间隔> <输出视频文件>"
#     exit 1
# fi

# # 获取参数
# input_video="$1"
# frame_interval="$2"
# output_video="$3"

input_video="/home/hongqingde/henanshifandaxue/DJI_0515.MP4"
frame_interval="22"
output_video="/home/hongqingde/hnsf_out/DJI_0515.MP4"

# 检查输入视频文件是否存在
if [ ! -f "$input_video" ]; then
    echo "输入视频文件不存在: $input_video"
    exit 1
fi

# 使用 ffmpeg 抽帧并生成新的视频
ffmpeg -i "$input_video" -vf "select=not(mod(n\,$frame_interval)),setpts=N/FRAME_RATE/TB" -af "aselect=not(mod(n\,$frame_interval)),asetpts=N/SR/TB" "$output_video"

# 检查命令是否成功执行
if [ $? -eq 0 ]; then
    echo "已成功抽帧并保存为: $output_video"
else
    echo "抽帧过程出错！"
    exit 1
fi
