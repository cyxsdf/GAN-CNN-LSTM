import torch
import os
import ctypes
import sys


def fix_cuda_issue():
    """尝试修复CUDA问题"""
    print("尝试修复CUDA问题...")

    # 方法1: 检查并加载必要的DLL
    try:
        ctypes.WinDLL('nvcuda.dll')
        print("✅ nvcuda.dll 加载成功")
    except Exception as e:
        print(f"❌ nvcuda.dll 加载失败: {e}")
        return False

    # 方法2: 设置环境变量
    cuda_path = os.environ.get('CUDA_PATH')
    if cuda_path:
        os.add_dll_directory(cuda_path + '\\bin')
        os.add_dll_directory(cuda_path + '\\libnvvp')
        print("✅ CUDA路径已添加到DLL搜索路径")

    # 方法3: 强制重新初始化
    try:
        if hasattr(torch.cuda, 'init'):
            torch.cuda.init()
            print("✅ CUDA强制初始化成功")
    except:
        print("⚠️  CUDA强制初始化失败")

    # 最终检查
    if torch.cuda.is_available():
        print("🎉 CUDA已恢复可用！")
        return True
    else:
        print("❌ CUDA仍然不可用")
        return False


# 运行修复
if fix_cuda_issue():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    print("使用CPU模式继续运行")

print(f"最终设备: {device}")