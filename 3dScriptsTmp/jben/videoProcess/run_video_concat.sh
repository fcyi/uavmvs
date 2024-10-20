#!/bin/bash

# 初始化变量
directory=""
output_file="mp4_files.txt"

# 解析命令行选项
while getopts "s:" opt; do
    case $opt in
        s)
            directory="$OPTARG"
            ;;
        *)
            echo "用法: $0 -s <文件夹路径>"
            exit 1
            ;;
    esac
done

# 检查是否指定了目录
if [ -z "$directory" ]; then
    echo "错误: 需要指定文件夹路径."
    echo "用法: $0 -s <文件夹路径>"
    exit 1
fi

# 检查目录是否存在
if [ ! -d "$directory" ]; then
    echo "目录不存在: $directory"
    exit 1
fi

# 清空输出文件，如果它已经存在
> "$directory/$output_file"

# # 查找所有 .mp4 文件，并将绝对路径格式化后保存到输出文件中
# find "$directory" -type f -name "*.mp4" -exec realpath {} \; | sed 's|^|file |' > "$directory/$output_file"

# 查找所有 .mp4, .wmv 和 .avi 文件，并将绝对路径格式化后保存到输出文件中，-iname可以不区分大小写
{
  find "$directory" -type f \( -iname "*.mp4" -o -iname "*.wmv" -o -iname "*.avi" \) -exec realpath {} \; | sort | sed 's|^|file |'
} > "$directory/$output_file"

# 检查是否找到了任何 .mp4 文件
if [ -s "$output_file" ]; then
    echo "找到的 .mp4 文件已保存到: $directory/$output_file"
else
    echo "没有找到 .mp4 文件."
    rm "$output_file"  # 如果没有找到文件，删除输出文件
fi

outVideo="merge_video.mp4"
ffmpeg -f concat -safe 0 -i "$directory/$output_file" -c copy -y "$directory/$outVideo"

