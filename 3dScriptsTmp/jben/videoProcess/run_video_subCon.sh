#!/bin/bash

# 初始化变量
inputDir=""
outputDir=""
intervalList=""
output_file="mp4_files.txt"

# 显示用法信息
usage() {
    echo "Usage: $0 -s <source_directory> -d <destination_directory> -i <intervalList>"
    exit 1
}

# 解析命令行选项
while getopts "s:o:i:" opt; do
    case $opt in
        s) inputDir="$OPTARG";;
        o) outputDir="$OPTARG";;
        i) intervalList="$OPTARG";;
        *)
            usage
            exit 1
            ;;
    esac
done

# 检查是否指定了目录，确保源目录和目标目录都已指定
if [ -z "${source_directory}" ] || [ -z "${destination_directory}" ] || [ -z "${intervalList}" ]; then
    usage
    exit 1
fi

# 检查目录是否存在
if [ ! -d "$inputDir" ]; then
    echo "目录不存在: $inputDir"
    exit 1
fi

if [ ! -d "$outputDir" ]; then
	# 创建输出目录（如果不存在）
	mkdir -p "$outputDir"
fi

# 查找所有视频文件并按字典序排序
video_files=$(find "$source_directory" -type f \( -iname "*.mp4" -o -iname "*.avi" -o -iname "*.wmv" \) | sort)


# 获取视频文件数量
num_videos=${#video_files[@]}

# 将间隔列表分割为数组
IFS=',' read -r -a intervals <<< "$intervalList"

# 检查间隔列表长度
if [ ${#intervals[@]} -eq 1 ]; then
    # 如果只有一个间隔，用该间隔处理所有视频
    interval=${intervals[0]}
else
    # 如果间隔数量与视频数量一致，分别处理
    if [ ${#intervals[@]} -ne $num_videos ]; then
        echo "Error: The length of the interval list must be 1 or equal to the number of videos."
        exit 1
    fi
fi

# 遍历每个视频文件进行抽帧
for ((i=0; i<num_videos; i++)); do
    video_file="${video_files[i]}"
    if [ -e "$video_file" ]; then
    	# 如果间隔数量大于1，则获取特定视频的间隔
		if [ ${#intervals[@]} -gt 1 ]; then
		    interval=${intervals[i]}
		fi

		# 获取视频文件名
		filename=$(basename "$video_file")
		
		# 检查文件扩展名
    	# extension="${filename##*.}"
    	
    	# 设置输出文件名 (抽帧后的文件名)    
        output_video="$outputDir/extracted_${file_name%.*}.mp4"
        
        # 使用 ffmpeg 抽帧并生成新的视频
        ffmpeg -i "$video_file" -vf "select=not(mod(n\,$frame_interval)),setpts=N/FRAME_RATE/TB" -af "aselect=not(mod(n\,$frame_interval)),asetpts=N/SR/TB" "$output_video"

        # 检查命令是否成功执行
        if [ $? -eq 0 ]; then
            echo "已成功抽帧: $output_video"
        else
            echo "抽帧过程出错: $video_file"
        fi
    fi
done

echo "处理完成！"


# 清空输出文件，如果它已经存在
> "$outputDir/$output_file"

# # 查找所有 .mp4 文件，并将绝对路径格式化后保存到输出文件中
# find "$inputDir" -type f -name "*.mp4" -exec realpath {} \; | sed 's|^|file |' > "$inputDir/$output_file"

# 查找所有 .mp4, .wmv 和 .avi 文件，并将绝对路径格式化后保存到输出文件中，-iname可以不区分大小写
{
  find "$outputDir" -type f \( -iname "*.mp4" -o -iname "*.wmv" -o -iname "*.avi" \) -exec realpath {} \; | sort | sed 's|^|file |'
} > "$outputDir/$output_file"

# 检查是否找到了任何 .mp4 文件
if [ -s "$output_file" ]; then
    echo "找到的 .mp4 文件已保存到: $outputDir/$output_file"
else
    echo "没有找到 .mp4 文件."
    rm "$output_file"  # 如果没有找到文件，删除输出文件
fi

outVideo="merge_video.mp4"
ffmpeg -f concat -safe 0 -i "$outputDir/$output_file" -c copy -y "$outputDir/$outVideo"

