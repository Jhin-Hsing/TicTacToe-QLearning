"""
演算法比較視覺化模組
比較不同 TD 演算法的學習效果
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from typing import List, Dict, Optional, Tuple
import sys
import os

# 添加專案根目錄到路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 設定中文字體
def setup_chinese_font():
    """設定中文字體"""
    chinese_fonts = [
        'Microsoft JhengHei',
        'Microsoft YaHei',
        'SimHei',
        'SimSun',
    ]
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    for font in chinese_fonts:
        if font in available_fonts:
            plt.rcParams['font.sans-serif'] = [font]
            plt.rcParams['axes.unicode_minus'] = False
            return
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

setup_chinese_font()

# 設定 seaborn 風格
sns.set_style("whitegrid")


class ComparisonPlotter:
    """演算法比較繪製器"""

    def __init__(self, figsize: Tuple[int, int] = (14, 10)):
        self.figsize = figsize
        self.colors = {
            'TD(0)': '#E74C3C',
            'SARSA': '#3498DB',
            'Q-Learning': '#27AE60'
        }
        setup_chinese_font()

    def compare_win_rates(
        self,
        agents: List,
        window_size: int = 100,
        title: str = "Algorithm Win Rate Comparison",
        save_path: Optional[str] = None,
        show: bool = False
    ):
        """比較多個演算法的勝率曲線"""
        fig, ax = plt.subplots(figsize=self.figsize)

        for agent in agents:
            rewards = agent.episode_rewards
            wins = [1 if r > 0 else 0 for r in rewards]
            win_rates = []

            for i in range(len(wins)):
                start_idx = max(0, i - window_size + 1)
                win_rate = sum(wins[start_idx:i+1]) / (i - start_idx + 1)
                win_rates.append(win_rate * 100)

            episodes = np.arange(1, len(wins) + 1)
            color = self.colors.get(agent.name, None)
            ax.plot(episodes, win_rates, linewidth=2, label=agent.name, color=color)

        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% Reference')

        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Win Rate (%)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylim([0, 100])
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"圖表已保存至：{save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def compare_rewards(
        self,
        agents: List,
        window_size: int = 100,
        title: str = "Algorithm Reward Comparison",
        save_path: Optional[str] = None,
        show: bool = False
    ):
        """比較多個演算法的累積獎勵曲線"""
        fig, ax = plt.subplots(figsize=self.figsize)

        for agent in agents:
            rewards = agent.episode_rewards
            episodes = np.arange(1, len(rewards) + 1)

            if len(rewards) >= window_size:
                moving_avg = np.convolve(rewards, np.ones(window_size) / window_size, mode='valid')
                color = self.colors.get(agent.name, None)
                ax.plot(episodes[window_size-1:], moving_avg,
                       linewidth=2, label=f'{agent.name} ({window_size}-Ep Avg)', color=color)

        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Cumulative Reward', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"圖表已保存至：{save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def compare_final_performance(
        self,
        eval_results: Dict[str, Dict],
        title: str = "Final Performance Comparison",
        save_path: Optional[str] = None,
        show: bool = False
    ):
        """比較最終評估結果"""
        fig, axes = plt.subplots(1, 2, figsize=self.figsize)

        agent_names = list(eval_results.keys())
        win_rates = [eval_results[name]['win_rate'] * 100 for name in agent_names]
        loss_rates = [eval_results[name]['loss_rate'] * 100 for name in agent_names]
        draw_rates = [eval_results[name]['draw_rate'] * 100 for name in agent_names]

        ax1 = axes[0]
        x = np.arange(len(agent_names))
        width = 0.25

        bars1 = ax1.bar(x - width, win_rates, width, label='Win', color='#27AE60')
        bars2 = ax1.bar(x, draw_rates, width, label='Draw', color='#F39C12')
        bars3 = ax1.bar(x + width, loss_rates, width, label='Loss', color='#E74C3C')

        ax1.set_xlabel('Algorithm', fontsize=12)
        ax1.set_ylabel('Rate (%)', fontsize=12)
        ax1.set_title('Win/Draw/Loss Rate Comparison', fontsize=12, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(agent_names)
        ax1.legend(loc='upper right')
        ax1.set_ylim([0, 100])
        ax1.grid(True, alpha=0.3, axis='y')

        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax1.annotate(f'{height:.1f}%',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)

        ax2 = axes[1]
        colors = ['#27AE60', '#F39C12', '#E74C3C']

        ax2.bar(agent_names, win_rates, label='Win', color=colors[0])
        ax2.bar(agent_names, draw_rates, bottom=win_rates, label='Draw', color=colors[1])
        ax2.bar(agent_names, loss_rates, bottom=[w+d for w, d in zip(win_rates, draw_rates)],
               label='Loss', color=colors[2])

        ax2.set_xlabel('Algorithm', fontsize=12)
        ax2.set_ylabel('Rate (%)', fontsize=12)
        ax2.set_title('Stacked Distribution', fontsize=12, fontweight='bold')
        ax2.legend(loc='upper right')
        ax2.set_ylim([0, 105])
        ax2.grid(True, alpha=0.3, axis='y')

        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"圖表已保存至：{save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def compare_convergence(
        self,
        agents: List,
        window_size: int = 100,
        threshold: float = 0.7,
        title: str = "Convergence Speed Comparison",
        save_path: Optional[str] = None,
        show: bool = False
    ):
        """比較演算法達到特定勝率的速度"""
        fig, ax = plt.subplots(figsize=self.figsize)

        convergence_points = {}

        for agent in agents:
            rewards = agent.episode_rewards
            wins = [1 if r > 0 else 0 for r in rewards]

            win_rates = []
            convergence_episode = None

            for i in range(len(wins)):
                start_idx = max(0, i - window_size + 1)
                win_rate = sum(wins[start_idx:i+1]) / (i - start_idx + 1)
                win_rates.append(win_rate)

                if win_rate >= threshold and convergence_episode is None:
                    convergence_episode = i + 1

            convergence_points[agent.name] = convergence_episode

            episodes = np.arange(1, len(wins) + 1)
            color = self.colors.get(agent.name, None)
            ax.plot(episodes, [w * 100 for w in win_rates],
                   linewidth=2, label=agent.name, color=color)

            if convergence_episode:
                ax.axvline(x=convergence_episode, color=color, linestyle='--', alpha=0.5)
                ax.scatter([convergence_episode], [threshold * 100], s=100, color=color, zorder=5)

        ax.axhline(y=threshold * 100, color='gray', linestyle='-', alpha=0.3,
                  label=f'Target ({threshold*100:.0f}%)')

        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Win Rate (%)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 100])

        info_text = "Convergence Episode:\n"
        for name, ep in convergence_points.items():
            if ep:
                info_text += f"  {name}: {ep}\n"
            else:
                info_text += f"  {name}: Not reached\n"

        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"圖表已保存至：{save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return convergence_points

    def plot_comprehensive_comparison(
        self,
        agents: List,
        eval_results: Dict[str, Dict],
        window_size: int = 100,
        save_path: Optional[str] = None,
        show: bool = False
    ):
        """繪製綜合比較圖"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. 勝率曲線比較
        ax1 = axes[0, 0]
        for agent in agents:
            rewards = agent.episode_rewards
            wins = [1 if r > 0 else 0 for r in rewards]
            win_rates = []
            for i in range(len(wins)):
                start_idx = max(0, i - window_size + 1)
                win_rate = sum(wins[start_idx:i+1]) / (i - start_idx + 1)
                win_rates.append(win_rate * 100)
            episodes = np.arange(1, len(wins) + 1)
            color = self.colors.get(agent.name, None)
            ax1.plot(episodes, win_rates, linewidth=2, label=agent.name, color=color)

        ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Win Rate (%)')
        ax1.set_title('Training Win Rate Comparison')
        ax1.legend(loc='lower right')
        ax1.set_ylim([0, 100])
        ax1.grid(True, alpha=0.3)

        # 2. 累積獎勵比較
        ax2 = axes[0, 1]
        for agent in agents:
            rewards = agent.episode_rewards
            episodes = np.arange(1, len(rewards) + 1)
            if len(rewards) >= window_size:
                moving_avg = np.convolve(rewards, np.ones(window_size) / window_size, mode='valid')
                color = self.colors.get(agent.name, None)
                ax2.plot(episodes[window_size-1:], moving_avg, linewidth=2,
                        label=agent.name, color=color)

        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Cumulative Reward')
        ax2.set_title('Reward Comparison (Moving Avg)')
        ax2.legend(loc='lower right')
        ax2.grid(True, alpha=0.3)

        # 3. 最終評估結果
        ax3 = axes[1, 0]
        agent_names = list(eval_results.keys())
        win_rates = [eval_results[name]['win_rate'] * 100 for name in agent_names]
        loss_rates = [eval_results[name]['loss_rate'] * 100 for name in agent_names]
        draw_rates = [eval_results[name]['draw_rate'] * 100 for name in agent_names]

        x = np.arange(len(agent_names))
        width = 0.25

        ax3.bar(x - width, win_rates, width, label='Win', color='#27AE60')
        ax3.bar(x, draw_rates, width, label='Draw', color='#F39C12')
        ax3.bar(x + width, loss_rates, width, label='Loss', color='#E74C3C')

        ax3.set_xlabel('Algorithm')
        ax3.set_ylabel('Rate (%)')
        ax3.set_title('Final Evaluation Results')
        ax3.set_xticks(x)
        ax3.set_xticklabels(agent_names)
        ax3.legend(loc='upper right')
        ax3.set_ylim([0, 100])
        ax3.grid(True, alpha=0.3, axis='y')

        # 4. 學習效率
        ax4 = axes[1, 1]
        stats = []
        for agent in agents:
            agent_stats = agent.get_stats()
            if 'num_state_action_pairs' in agent_stats:
                stats.append((agent.name, agent_stats['num_state_action_pairs']))
            elif 'num_states' in agent_stats:
                stats.append((agent.name, agent_stats['num_states']))

        if stats:
            names, counts = zip(*stats)
            bars = ax4.bar(names, counts, color=[self.colors.get(n, 'gray') for n in names])
            ax4.set_xlabel('Algorithm')
            ax4.set_ylabel('State-Action Pairs Learned')
            ax4.set_title('Learning Complexity')
            ax4.grid(True, alpha=0.3, axis='y')

            for bar, count in zip(bars, counts):
                ax4.annotate(f'{count}',
                           xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=10)

        plt.suptitle('TD Algorithm Comprehensive Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"圖表已保存至：{save_path}")

        if show:
            plt.show()
        else:
            plt.close()
