"""
遊戲視覺化模組
顯示井字棋遊戲過程的動畫和棋盤狀態
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from typing import List, Tuple, Optional
import sys
import os

# 添加專案根目錄到路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.tictactoe import TicTacToe


class GameVisualizer:
    """遊戲視覺化器"""

    def __init__(self, figsize: Tuple[int, int] = (8, 8)):
        """
        初始化視覺化器

        Args:
            figsize: 圖形大小
        """
        self.figsize = figsize
        self.colors = {
            'background': '#F0E6D3',
            'grid': '#2C3E50',
            'X': '#E74C3C',
            'O': '#3498DB',
            'win_line': '#27AE60'
        }

    def draw_board(
        self,
        ax: plt.Axes,
        board: np.ndarray,
        title: str = "",
        highlight_last: Optional[Tuple[int, int]] = None,
        win_line: Optional[List[Tuple[int, int]]] = None
    ):
        """
        繪製棋盤

        Args:
            ax: matplotlib axes
            board: 棋盤狀態
            title: 標題
            highlight_last: 高亮最後一步的位置
            win_line: 獲勝連線的位置
        """
        ax.clear()
        ax.set_xlim(-0.1, 3.1)
        ax.set_ylim(-0.1, 3.1)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, fontsize=16, fontweight='bold', fontname='Microsoft JhengHei')

        # 繪製背景
        background = patches.Rectangle((0, 0), 3, 3, linewidth=0,
                                         facecolor=self.colors['background'])
        ax.add_patch(background)

        # 繪製格線
        for i in range(1, 3):
            ax.axhline(y=i, color=self.colors['grid'], linewidth=3, xmin=0, xmax=1)
            ax.axvline(x=i, color=self.colors['grid'], linewidth=3, ymin=0, ymax=1)

        # 繪製棋子
        for i in range(3):
            for j in range(3):
                x = j + 0.5
                y = 2.5 - i  # 翻轉 y 座標

                if board[i, j] == 1:  # X
                    self._draw_x(ax, x, y, highlight=(highlight_last == (i, j)))
                elif board[i, j] == -1:  # O
                    self._draw_o(ax, x, y, highlight=(highlight_last == (i, j)))

        # 繪製獲勝連線
        if win_line:
            self._draw_win_line(ax, win_line)

    def _draw_x(self, ax: plt.Axes, x: float, y: float, highlight: bool = False):
        """繪製 X 棋子"""
        color = self.colors['X']
        linewidth = 8 if highlight else 6
        offset = 0.3

        ax.plot([x - offset, x + offset], [y - offset, y + offset],
                color=color, linewidth=linewidth, solid_capstyle='round')
        ax.plot([x - offset, x + offset], [y + offset, y - offset],
                color=color, linewidth=linewidth, solid_capstyle='round')

        if highlight:
            circle = plt.Circle((x, y), 0.45, fill=False, color=color,
                                linewidth=2, linestyle='--', alpha=0.5)
            ax.add_patch(circle)

    def _draw_o(self, ax: plt.Axes, x: float, y: float, highlight: bool = False):
        """繪製 O 棋子"""
        color = self.colors['O']
        linewidth = 8 if highlight else 6

        circle = plt.Circle((x, y), 0.3, fill=False, color=color, linewidth=linewidth)
        ax.add_patch(circle)

        if highlight:
            outer_circle = plt.Circle((x, y), 0.45, fill=False, color=color,
                                       linewidth=2, linestyle='--', alpha=0.5)
            ax.add_patch(outer_circle)

    def _draw_win_line(self, ax: plt.Axes, positions: List[Tuple[int, int]]):
        """繪製獲勝連線"""
        if len(positions) < 2:
            return

        # 轉換座標
        coords = [(j + 0.5, 2.5 - i) for i, j in positions]
        x_coords = [c[0] for c in coords]
        y_coords = [c[1] for c in coords]

        ax.plot(x_coords, y_coords, color=self.colors['win_line'],
                linewidth=10, alpha=0.6, solid_capstyle='round')

    def visualize_game(
        self,
        game_history: List[Tuple[np.ndarray, Tuple[int, int], int]],
        interval: int = 1000,
        save_path: Optional[str] = None
    ):
        """
        視覺化完整的遊戲過程

        Args:
            game_history: 遊戲歷史 [(棋盤狀態, 動作, 玩家), ...]
            interval: 動畫間隔（毫秒）
            save_path: 保存路徑（可選）
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        def update(frame):
            if frame < len(game_history):
                board, action, player = game_history[frame]
                player_name = "X (AI)" if player == 1 else "O (對手)"
                title = f"步驟 {frame + 1}: {player_name} 下在 {action}"
                self.draw_board(ax, board, title, highlight_last=action)
            return []

        anim = FuncAnimation(fig, update, frames=len(game_history),
                            interval=interval, repeat=False)

        if save_path:
            anim.save(save_path, writer='pillow', fps=1)
            print(f"動畫已保存至：{save_path}")

        plt.tight_layout()
        plt.show()

    def show_final_board(
        self,
        board: np.ndarray,
        winner: Optional[int] = None,
        title: str = "遊戲結果"
    ):
        """
        顯示最終棋盤狀態

        Args:
            board: 最終棋盤狀態
            winner: 獲勝者 (1, -1, 或 None 表示平局)
            title: 標題
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # 找出獲勝連線
        win_line = self._find_win_line(board) if winner else None

        # 設定標題
        if winner == 1:
            result_text = "X (AI) 獲勝！"
        elif winner == -1:
            result_text = "O (對手) 獲勝！"
        else:
            result_text = "平局！"

        full_title = f"{title}\n{result_text}"
        self.draw_board(ax, board, full_title, win_line=win_line)

        plt.tight_layout()
        plt.show()

    def _find_win_line(self, board: np.ndarray) -> Optional[List[Tuple[int, int]]]:
        """找出獲勝連線"""
        # 檢查行
        for i in range(3):
            if abs(board[i, :].sum()) == 3:
                return [(i, 0), (i, 1), (i, 2)]

        # 檢查列
        for j in range(3):
            if abs(board[:, j].sum()) == 3:
                return [(0, j), (1, j), (2, j)]

        # 檢查對角線
        if abs(board.diagonal().sum()) == 3:
            return [(0, 0), (1, 1), (2, 2)]

        # 檢查反對角線
        if abs(np.fliplr(board).diagonal().sum()) == 3:
            return [(0, 2), (1, 1), (2, 0)]

        return None

    def show_multiple_games(
        self,
        games: List[Tuple[np.ndarray, int]],
        cols: int = 4,
        title: str = "多場遊戲結果"
    ):
        """
        顯示多場遊戲的最終結果

        Args:
            games: 遊戲列表 [(最終棋盤, 獲勝者), ...]
            cols: 每行顯示的遊戲數
            title: 整體標題
        """
        n_games = len(games)
        rows = (n_games + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        fig.suptitle(title, fontsize=16, fontweight='bold', fontname='Microsoft JhengHei')

        # 確保 axes 是二維的
        if rows == 1:
            axes = axes.reshape(1, -1)
        if cols == 1:
            axes = axes.reshape(-1, 1)

        for idx, (board, winner) in enumerate(games):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col]

            win_line = self._find_win_line(board) if winner else None

            if winner == 1:
                result = "X 贏"
            elif winner == -1:
                result = "O 贏"
            else:
                result = "平局"

            self.draw_board(ax, board, f"遊戲 {idx + 1}: {result}", win_line=win_line)

        # 隱藏多餘的子圖
        for idx in range(n_games, rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row, col].axis('off')

        plt.tight_layout()
        plt.show()


def record_game(env: TicTacToe, agent, opponent) -> List[Tuple[np.ndarray, Tuple[int, int], int]]:
    """
    記錄一場遊戲的過程

    Args:
        env: 遊戲環境
        agent: AI Agent
        opponent: 對手

    Returns:
        遊戲歷史
    """
    history = []
    state = env.reset()
    done = False

    # 保存原始探索率
    original_epsilon = getattr(agent, 'epsilon', 0)
    if hasattr(agent, 'epsilon'):
        agent.epsilon = 0  # 評估時不探索

    while not done:
        if env.current_player == 1:  # AI 的回合
            legal_actions = env.get_legal_actions()

            # 根據 agent 類型選擇動作
            if hasattr(agent, 'choose_action'):
                try:
                    action = agent.choose_action(state, legal_actions, env.clone)
                except TypeError:
                    action = agent.choose_action(state, legal_actions)
            else:
                action = legal_actions[0]

            state, _, done, info = env.step(action)
            history.append((state.copy(), action, 1))

        else:  # 對手回合
            legal_actions = env.get_legal_actions()
            action = opponent.choose_action(legal_actions)
            state, _, done, info = env.step(action)
            history.append((state.copy(), action, -1))

    # 恢復探索率
    if hasattr(agent, 'epsilon'):
        agent.epsilon = original_epsilon

    return history


if __name__ == "__main__":
    # 測試視覺化
    from opponents.random_player import RandomPlayer
    from agents.qlearning_agent import QLearningAgent

    env = TicTacToe()
    visualizer = GameVisualizer()

    # 創建一個簡單的測試棋盤
    test_board = np.array([
        [1, -1, 1],
        [0, 1, -1],
        [-1, 0, 1]
    ])

    visualizer.show_final_board(test_board, winner=1, title="測試棋盤")
