"""
輔助函數模組
提供各種工具函數
"""
import numpy as np
import os
from typing import Optional


def create_output_dir(dir_name: str = "output") -> str:
    """
    創建輸出目錄

    Args:
        dir_name: 目錄名稱

    Returns:
        目錄路徑
    """
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = os.path.join(base_path, dir_name)

    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"已創建目錄：{output_path}")

    return output_path


def save_results_to_file(
    results: dict,
    filename: str,
    output_dir: Optional[str] = None
):
    """
    將結果保存到文字檔案

    Args:
        results: 結果字典
        filename: 檔案名稱
        output_dir: 輸出目錄
    """
    if output_dir is None:
        output_dir = create_output_dir()

    filepath = os.path.join(output_dir, filename)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("強化學習井字棋專題 - 訓練結果報告\n")
        f.write("=" * 60 + "\n\n")

        for key, value in results.items():
            if isinstance(value, dict):
                f.write(f"\n【{key}】\n")
                f.write("-" * 40 + "\n")
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, float):
                        f.write(f"  {sub_key}: {sub_value:.4f}\n")
                    else:
                        f.write(f"  {sub_key}: {sub_value}\n")
            else:
                f.write(f"{key}: {value}\n")

    print(f"結果已保存至：{filepath}")


def print_board(board: np.ndarray):
    """
    在終端機中打印棋盤

    Args:
        board: 棋盤狀態
    """
    symbols = {0: '.', 1: 'X', -1: 'O'}

    print("\n  0 1 2")
    for i in range(3):
        print(f"{i} ", end="")
        for j in range(3):
            print(symbols[board[i, j]], end=" ")
        print()
    print()


def calculate_statistics(data: list) -> dict:
    """
    計算統計數據

    Args:
        data: 數據列表

    Returns:
        統計結果字典
    """
    if not data:
        return {
            'mean': 0,
            'std': 0,
            'min': 0,
            'max': 0,
            'count': 0
        }

    return {
        'mean': np.mean(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data),
        'count': len(data)
    }


def format_time(seconds: float) -> str:
    """
    格式化時間

    Args:
        seconds: 秒數

    Returns:
        格式化的時間字串
    """
    if seconds < 60:
        return f"{seconds:.2f} 秒"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{int(minutes)} 分 {secs:.2f} 秒"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{int(hours)} 小時 {int(minutes)} 分 {secs:.2f} 秒"
