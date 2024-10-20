#!/bin/bash

# 获取当前Git项目的根目录路径
git_root=$(git rev-parse --show-toplevel)

# 遍历Git项目中的每个文件
modified_files=0
total_files=0
while IFS= read -r -d '' file; do
  # 检查文件的权限是否有修改  
  git_diff=$(git diff "$file")
  
  if [[ $git_diff == *"old mode"* && $git_diff == *"new mode"* ]]; then
    # 文件权限被修改，还原相应的权限
    #echo "Restoring permissions for $file"
    #git checkout -- "$file"
    echo "文件权限已修改：$file"
    echo "权限修改情况："
    echo "$git_diff"
    od=$(echo "$git_diff" | grep "^old mode" | awk '{print $3}')
    nw=$(echo "$git_diff" | grep "^new mode" | awk '{print $3}')
    echo "old mode: $od"
    echo "new mode: $nw"
    # 转换权限表示形式
	od_perms=$(printf "%04o\n" $((8#${od:1})))
	nw_perms=$(printf "%04o\n" $((8#${nw:1})))
    # 打印转换后的权限
	echo "old mode: $od_perms"
	echo "new mode: $nw_perms"
	chmod "$od_perms" "$file"
    ((modified_files++))
  fi
  
  ((total_files++))

done < <(find "$git_root" -type f -print0)

echo "total files is $total_files"
if [[ $modified_files -gt 0 ]]; then
  echo "File permissions have been restored for $modified_files files. total files is $total_files"
else
  echo "No file permissions modifications found in the Git repository."
fi
