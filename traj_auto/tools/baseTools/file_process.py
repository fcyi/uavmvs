import subprocess
import sys
import os
import stat
import shutil


def set_write_permission(filePath_):
    # 检查文件权限
    fileStat_ = os.stat(filePath_)

    # 判断是否为只读（即没有写入权限）
    if not fileStat_.st_mode & stat.S_IWUSR & stat.S_IRUSR & stat.S_IRGRP & stat.S_IROTH:
        # 赋予写入权限
        os.chmod(filePath_, fileStat_.st_mode | stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
        print(f"{filePath_} 已成功赋予写入权限。")
    else:
        pass


def add_write_permission_to_files(directory_):
    # 遍历指定目录
    for root_, dirs_, files_ in os.walk(directory_):
        # 更改当前目录的权限
        currentPermissions_ = stat.S_IMODE(os.stat(root_).st_mode)
        os.chmod(root_, currentPermissions_ | stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)  # 777 权限
        for file_ in files_:
            filePath_ = os.path.join(root_, file_)
            # 获取当前文件的权限
            currentPermissions_ = stat.S_IMODE(os.stat(filePath_).st_mode)
            # 添加写入权限
            newPermissions_ = currentPermissions_ | stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH  # 给拥有者添加写入权限
            os.chmod(filePath_, newPermissions_)  # 修改权限
        for dir_ in dirs_:
            dirPath_ = os.path.join(root_, dir_)
            # 获取当前文件的权限
            currentPermissions_ = stat.S_IMODE(os.stat(dirPath_).st_mode)
            # 添加写入权限
            newPermissions_ = currentPermissions_ | stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH  # 给拥有者添加写入权限
            os.chmod(dirPath_, newPermissions_)  # 修改权限


def add_write_permission_to_files_chmod(directory_):
    try:
        # 使用 subprocess 调用 chmod 命令，不带通配符
        subprocess.run(['chmod', '-R', '777', directory_], check=True)
        print(f"Permissions changed to 777 for all files and directories in {directory_}.")
    except subprocess.CalledProcessError as e_:
        print(f"Error occurred: {e_}")


def remove_directory_chmod(directory_):
    try:
        if os.path.exists(directory_) and os.path.isdir(directory_):
            shutil.rmtree(directory_)  # 删除整个文件夹及其内容
            print(f"'{directory_}' have beed removed. ")
    except subprocess.CalledProcessError as e_:
        print(f"Error occurred: {e_}")
