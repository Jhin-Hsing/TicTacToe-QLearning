"""
學習曲線視覺化模組
繪製訓練過程中的各種指標曲線
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from typing import List, Dict, Optional, Tuple
import sys
import os

# 添加專案根目錄到路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 設定中文字體 - 使用系統可用的中文字體
def setup_chinese_font():
    """設定中文字體"""
    import matplotlib.font_manager as fm

    # Windows 常見中文字體
    chinese_fonts = [
        'Microsoft JhengHei',  # 微軟正黑體
        'Microsoft YaHei',     # 微軟雅黑
        'SimHei',              # 黑體
        'SimSun',              # 宋體
        'KaiTi',               # 楷體
        'FangSong',            # 仿宋
    ]

    # 獲取系統可用字體
    available_fonts = [f.name for f in fm.fontManager.ttflist]

    # 找到第一個可用的中文字體
    for font in chinese_fonts:
        if font in available_fonts:
            plt.rcParams['font.sans-serif'] = [font]
            plt.rcParams['axes.unicode_minus'] = False
            return font

    # 如果找不到，嘗試直接設定
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    return None

# 初始化字體設定
setup_chinese_font()

# 設定 seaborn 風格
sns.set_style("whitegrid")
sns.set_palette("husl")


class LearningCurvePlotter:
    """學習曲線繪製器"""

    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        初始化繪製器

        Args:
            figsize: 圖形大小
        """
        self.figsize = figsize
        setup_chinese_font()

    def plot_rewards(
        self,
        rewards: List[float],
        window_size: int = 100,
        title: str = "累積獎勵曲線",
        agent_name: str = "Agent",
        save_path: Optional[str] = None,
        show: bool = False
    ):
        """
        繪製累積獎勵曲線

        Args:
            rewards: 每個 episode 的總獎勵
            window_size: 移動平均窗口大小
            title: 圖表標題
            agent_name: Agent 名稱
            save_path: 保存路徑
            show: 是否顯示圖表
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        episodes = np.arange(1, len(rewards) + 1)

        # 原始數據（半透明）
        ax.plot(episodes, rewards, alpha=0.3, label='Raw Reward')

        # 移動平均
        if len(rewards) >= window_size:
            moving_avg = self._moving_average(rewards, window_size)
            ax.plot(episodes[window_size-1:], moving_avg,
                   linewidth=2, label=f'{window_size}-Episode Moving Avg')

        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Cumulative Reward', fontsize=12)
        ax.set_title(f'{title} - {agent_name}', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"圖表已保存至：{save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_win_rate(
        self,
        rewards: List[float],
        window_size: int = 100,
        title: str = "勝率曲線",
        agent_name: str = "Agent",
        save_path: Optional[str] = None,
        show: bool = False
    ):
        """
        繪製勝率曲線
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # 計算滾動勝率
        wins = [1 if r > 0 else 0 for r in rewards]
        win_rates = []

        for i in range(len(wins)):
            start_idx = max(0, i - window_size + 1)
            win_rate = sum(wins[start_idx:i+1]) / (i - start_idx + 1)
            win_rates.append(win_rate * 100)

        episodes = np.arange(1, len(wins) + 1)

        ax.plot(episodes, win_rates, linewidth=2, color='green')
        ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% Reference')

        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Win Rate (%)', fontsize=12)
        ax.set_title(f'{title} - {agent_name}', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 100])
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"圖表已保存至：{save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_epsilon_decay(
        self,
        epsilon_history: List[float],
        title: str = "探索率衰減曲線",
        agent_name: str = "Agent",
        save_path: Optional[str] = None,
        show: bool = False
    ):
        """繪製探索率衰減曲線"""
        fig, ax = plt.subplots(figsize=self.figsize)

        episodes = np.arange(1, len(epsilon_history) + 1)

        ax.plot(episodes, epsilon_history, linewidth=2, color='purple')

        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Epsilon', fontsize=12)
        ax.set_title(f'{title} - {agent_name}', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"圖表已保存至：{save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_episode_lengths(
        self,
        lengths: List[int],
        window_size: int = 100,
        title: str = "遊戲長度曲線",
        agent_name: str = "Agent",
        save_path: Optional[str] = None,
        show: bool = False
    ):
        """繪製每個 episode 的遊戲長度"""
        fig, ax = plt.subplots(figsize=self.figsize)

        episodes = np.arange(1, len(lengths) + 1)

        ax.plot(episodes, lengths, alpha=0.3, label='Raw Length')

        if len(lengths) >= window_size:
            moving_avg = self._moving_average(lengths, window_size)
            ax.plot(episodes[window_size-1:], moving_avg,
                   linewidth=2, label=f'{window_size}-Episode Moving Avg')

        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Game Steps', fontsize=12)
        ax.set_title(f'{title} - {agent_name}', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"圖表已保存至：{save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_all_metrics(
        self,
        agent,
        window_size: int = 100,
        title_prefix: str = "",
        save_path: Optional[str] = None,
        show: bool = False
    ):
        """
        繪製所有學習指標

        Args:
            agent: 訓練好的 Agent
            window_size: 移動平均窗口大小
            title_prefix: 標題前綴
            save_path: 保存路徑
            show: 是否顯示圖表
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        agent_name = agent.name
        rewards = agent.episode_rewards
        lengths = agent.episode_lengths
        epsilons = agent.epsilon_history

        episodes = np.arange(1, len(rewards) + 1)

        # 1. 累積獎勵
        ax1 = axes[0, 0]
        ax1.plot(episodes, rewards, alpha=0.3, label='Raw Reward')
        if len(rewards) >= window_size:
            moving_avg = self._moving_average(rewards, window_size)
            ax1.plot(episodes[window_size-1:], moving_avg,
                    linewidth=2, label=f'{window_size}-Ep Moving Avg')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Cumulative Reward')
        ax1.set_title('Cumulative Reward Curve')
        ax1.legend(loc='lower right')
        ax1.grid(True, alpha=0.3)

        # 2. 勝率
        ax2 = axes[0, 1]
        wins = [1 if r > 0 else 0 for r in rewards]
        win_rates = []
        for i in range(len(wins)):
            start_idx = max(0, i - window_size + 1)
            win_rate = sum(wins[start_idx:i+1]) / (i - start_idx + 1)
            win_rates.append(win_rate * 100)
        ax2.plot(episodes, win_rates, linewidth=2, color='green')
        ax2.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% Reference')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Win Rate (%)')
        ax2.set_title('Win Rate Curve')
        ax2.set_ylim([0, 100])
        ax2.legend(loc='lower right')
        ax2.grid(True, alpha=0.3)

        # 3. 探索率衰減
        ax3 = axes[1, 0]
        ax3.plot(episodes, epsilons, linewidth=2, color='purple')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Epsilon')
        ax3.set_title('Epsilon Decay Curve')
        ax3.set_ylim([0, 1.05])
        ax3.grid(True, alpha=0.3)

        # 4. 遊戲長度
        ax4 = axes[1, 1]
        ax4.plot(episodes, lengths, alpha=0.3, label='Raw Length')
        if len(lengths) >= window_size:
            moving_avg = self._moving_average(lengths, window_size)
            ax4.plot(episodes[window_size-1:], moving_avg,
                    linewidth=2, label=f'{window_size}-Ep Moving Avg')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Game Steps')
        ax4.set_title('Episode Length Curve')
        ax4.legend(loc='upper right')
        ax4.grid(True, alpha=0.3)

        plt.suptitle(f'{title_prefix}{agent_name} Learning Curves',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"圖表已保存至：{save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def _moving_average(self, data: List[float], window_size: int) -> np.ndarray:
        """計算移動平均"""
        return np.convolve(data, np.ones(window_size) / window_size, mode='valid')
