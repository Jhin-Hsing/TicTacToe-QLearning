"""
井字棋圖形化介面
提供：
1. 動態展示 AI vs AI 對戰動畫
2. 人類玩家 vs AI 對戰介面
"""
import tkinter as tk
from tkinter import messagebox, ttk
import numpy as np
import os
import sys
import time
from typing import Optional

# 添加專案路徑
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment.tictactoe import TicTacToe
from agents.td0_agent import TD0Agent
from agents.sarsa_agent import SARSAAgent
from agents.qlearning_agent import QLearningAgent
from opponents.random_player import RandomPlayer


def resource_path(relative_path):
    """ 獲取資源絕對路徑，兼容開發環境與 PyInstaller 打包環境 """
    try:
        # PyInstaller 創建的暫存資料夾
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.dirname(os.path.abspath(__file__))

    return os.path.join(base_path, relative_path)


class TicTacToeGUI:
    """井字棋圖形介面"""

    def __init__(self, master: tk.Tk):
        self.master = master
        self.master.title("井字棋 - 強化學習專題")
        self.master.resizable(False, False)

        # 遊戲相關
        self.env = TicTacToe()
        self.agent = None
        self.opponent = None
        self.game_mode = None  # 'human_vs_ai' or 'ai_vs_ai'
        self.human_player = None  # 1 (X, 先手) or -1 (O, 後手)
        self.is_game_running = False
        self.animation_speed = 1000  # 毫秒

        # 載入模型
        self.load_agents()

        # 建立介面
        self.create_widgets()

        # 設定視窗位置
        self.center_window()

    def center_window(self):
        """將視窗置中"""
        self.master.update_idletasks()
        width = self.master.winfo_width()
        height = self.master.winfo_height()
        x = (self.master.winfo_screenwidth() // 2) - (width // 2)
        y = (self.master.winfo_screenheight() // 2) - (height // 2)
        self.master.geometry(f'+{x}+{y}')

    def load_agents(self):
        """載入訓練好的 TD(0) 模型"""
        # 使用 resource_path 確保在 exe 中也能找到 models 資料夾
        models_dir = resource_path('models')
        self.agents = {}
        
        filepath = os.path.join(models_dir, 'td0_agent.pkl')
        if os.path.exists(filepath):
            try:
                agent = TD0Agent()
                agent.load(filepath)
                agent.epsilon = 0  # 遊戲時不探索
                self.agents['TD(0) AI'] = agent
                print(f"已載入 TD(0) 模型 (Q-table size: {len(agent.Q)})")
            except Exception as e:
                print(f"載入 TD(0) 失敗: {e}")

        if not self.agents:
            print("警告：找不到模型，使用未訓練的 Agent")
            self.agents['TD(0) (未訓練)'] = TD0Agent(epsilon=0.1)

    def create_widgets(self):
        """建立美化後的介面元件"""
        # 主框架
        self.main_frame = tk.Frame(self.master, padx=30, pady=30, bg='#F8F9FA')
        self.main_frame.pack(expand=True, fill=tk.BOTH)

        # 標題
        title_label = tk.Label(self.main_frame, text="井字棋 AI 對戰",
                              font=('Microsoft JhengHei', 24, 'bold'),
                              bg='#F8F9FA', fg='#2C3E50')
        title_label.pack(pady=(0, 20))

        # 內容區域
        content_frame = tk.Frame(self.main_frame, bg='#F8F9FA')
        content_frame.pack()

        # 左側：棋盤
        board_container = tk.Frame(content_frame, bg='#BDC3C7', padx=3, pady=3)
        board_container.pack(side=tk.LEFT, padx=(0, 30))

        self.buttons = []
        for i in range(3):
            row = []
            for j in range(3):
                btn = tk.Button(
                    board_container,
                    text='',
                    font=('Segoe UI', 42, 'bold'),
                    width=3,
                    height=1,
                    bg='white',
                    activebackground='#F1F2F6',
                    relief=tk.FLAT,
                    bd=0,
                    command=lambda r=i, c=j: self.on_cell_click(r, c)
                )
                btn.grid(row=i, column=j, padx=2, pady=2)
                row.append(btn)
            self.buttons.append(row)

        # 右側：控制面板
        control_frame = tk.Frame(content_frame, bg='#F8F9FA')
        control_frame.pack(side=tk.LEFT, fill=tk.Y)

        # 遊戲設定組
        settings_group = tk.LabelFrame(control_frame, text=" 遊戲設定 ",
                                      font=('Microsoft JhengHei', 11, 'bold'),
                                      bg='#F8F9FA', padx=15, pady=15)
        settings_group.pack(fill=tk.X, pady=(0, 20))

        self.turn_var = tk.StringVar(value='first')
        tk.Radiobutton(settings_group, text="我先手 (O)", variable=self.turn_var,
                       value='first', bg='#F8F9FA', font=('Microsoft JhengHei', 10)).pack(anchor=tk.W)
        tk.Radiobutton(settings_group, text="AI 先手 (X)", variable=self.turn_var,
                       value='second', bg='#F8F9FA', font=('Microsoft JhengHei', 10)).pack(anchor=tk.W)

        # 操作按鈕
        self.start_btn = tk.Button(control_frame, text="開始遊戲",
                                   font=('Microsoft JhengHei', 12, 'bold'),
                                   bg='#3498DB', fg='white',
                                   activebackground='#2980B9',
                                   activeforeground='white',
                                   relief=tk.FLAT,
                                   command=self.start_game,
                                   width=15, pady=10)
        self.start_btn.pack(pady=5)

        self.reset_btn = tk.Button(control_frame, text="重新開始",
                                   font=('Microsoft JhengHei', 11),
                                   bg='#ECF0F1', fg='#2C3E50',
                                   relief=tk.FLAT,
                                   command=self.reset_game,
                                   width=15, pady=5)
        self.reset_btn.pack(pady=5)

        # 狀態顯示
        self.status_var = tk.StringVar(value="準備就緒")
        self.status_label = tk.Label(self.main_frame, textvariable=self.status_var,
                                   font=('Microsoft JhengHei', 16),
                                   bg='#F8F9FA', fg='#34495E')
        self.status_label.pack(pady=(25, 0))

    def start_game(self):
        """開始新遊戲"""
        self.reset_board()
        
        # 固定使用 TD(0) 模型
        if 'TD(0) AI' in self.agents:
            self.agent = self.agents['TD(0) AI']
        else:
            self.agent = TD0Agent(epsilon=0)

        self.is_game_running = True
        self.game_mode = 'human_vs_ai'

        if self.turn_var.get() == 'first':
            self.human_player = 1  # 人類先手 (O) - 注意：為了視覺一致性，我們內部交換符號
            # 在 TicTacToe 環境中 1 是 X，-1 是 O。
            # 我們讓人類先手時，人類是環境的 1 (X)
            self.status_var.set("輪到你了")
            self.status_label.config(fg='#3498DB')
        else:
            self.human_player = -1 # 人類後手 (O)
            self.status_var.set("AI 思考中...")
            self.status_label.config(fg='#E74C3C')
            self.master.after(600, self.ai_move)

    def make_move(self, row: int, col: int):
        """執行一步棋並更新 UI"""
        current_player = self.env.current_player
        self.env.step((row, col))

        # 美化標記
        if current_player == 1: # 先手 (通常代表進攻方)
            self.buttons[row][col].config(text='X', fg='#E74C3C', disabledforeground='#E74C3C')
        else: # 後手
            self.buttons[row][col].config(text='O', fg='#3498DB', disabledforeground='#3498DB')

        self.buttons[row][col].config(state=tk.DISABLED)

    def on_cell_click(self, row: int, col: int):
        """處理點擊"""
        if not self.is_game_running or self.env.current_player != self.human_player:
            return

        if (row, col) not in self.env.get_legal_actions():
            return

        self.make_move(row, col)

        if self.env.done:
            self.show_game_result()
        else:
            self.status_var.set("AI 思考中...")
            self.status_label.config(fg='#E74C3C')
            self.master.after(500, self.ai_move)

    def ai_move(self):
        """AI 決策與執行"""
        if not self.is_game_running or self.env.done:
            return

        legal_actions = self.env.get_legal_actions()
        state = self.env.board.copy()
        
        # 這裡不需 clone 函數，直接傳入狀態與合法動作
        action = self.agent.choose_action(state, legal_actions)

        self.make_move(action[0], action[1])

        if self.env.done:
            self.show_game_result()
        else:
            self.status_var.set("輪到你了")
            self.status_label.config(fg='#3498DB')

    def reset_game(self):
        """重置遊戲"""
        self.is_game_running = False
        self.reset_board()
        self.status_var.set("準備就緒")
        self.status_label.config(fg='#34495E')

    def reset_board(self):
        """重置棋盤"""
        self.env.reset()
        for i in range(3):
            for j in range(3):
                self.buttons[i][j].config(text='', bg='white', state=tk.NORMAL)

    def show_game_result(self):
        """顯示遊戲結果"""
        self.is_game_running = False

        # 高亮獲勝連線
        win_line = self.find_win_line()
        if win_line:
            for row, col in win_line:
                self.buttons[row][col].config(bg='#F1C40F')

        # 顯示結果
        winner = self.env.winner
        if winner == self.human_player:
            result = "恭喜！你贏了！🎉"
            self.status_label.config(fg='#27AE60')
        elif winner == -self.human_player:
            result = "AI 獲勝！再接再厲！"
            self.status_label.config(fg='#E74C3C')
        else:
            result = "平局！勢均力敵！"
            self.status_label.config(fg='#7F8C8D')

        self.status_var.set(result)

        # 延遲顯示訊息框，避免阻擋最後一步的視覺更新
        self.master.after(200, lambda: messagebox.showinfo("遊戲結束", result))

    def find_win_line(self):
        """找出獲勝連線"""
        board = self.env.board
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


    def ai_vs_ai_step(self):
        """AI vs AI 一步"""
        if not self.is_game_running or self.env.done:
            return

        current_player = self.env.current_player
        legal_actions = self.env.get_legal_actions()

        if not legal_actions:
            return

        state = self.env.board.copy()

        if current_player == 1:
            # 主要 AI (X)
            if isinstance(self.agent, TD0Agent):
                action = self.agent.choose_action(state, legal_actions, self.env.clone)
            else:
                action = self.agent.choose_action(state, legal_actions)
            player_name = f"AI ({self.ai_var.get()})"
        else:
            # 對手 (O) - 隨機
            action = self.opponent.choose_action(legal_actions)
            player_name = "Random AI"

        # 執行動作
        self.make_move(action[0], action[1])
        self.status_var.set(f"{player_name} 下在 ({action[0]}, {action[1]})")

        if self.env.done:
            self.show_game_result()
        else:
            self.master.after(self.animation_speed, self.ai_vs_ai_step)

    def make_move(self, row: int, col: int):
        """執行一步棋"""
        current_player = self.env.current_player

        # 更新環境
        self.env.step((row, col))

        # 更新按鈕
        if current_player == 1:
            self.buttons[row][col].config(text='X', fg='#E74C3C')
        else:
            self.buttons[row][col].config(text='O', fg='#3498DB')

        self.buttons[row][col].config(state=tk.DISABLED)

    def show_game_result(self):
        """顯示遊戲結果"""
        self.is_game_running = False

        # 高亮獲勝連線
        win_line = self.find_win_line()
        if win_line:
            for row, col in win_line:
                self.buttons[row][col].config(bg='#F1C40F')

        # 顯示結果
        winner = self.env.winner
        if self.game_mode == 'human_vs_ai':
            if winner == self.human_player:
                result = "恭喜！你贏了！🎉"
            elif winner == -self.human_player:
                result = "AI 獲勝！再接再厲！"
            else:
                result = "平局！勢均力敵！"
        else:
            if winner == 1:
                result = f"{self.ai_var.get()} (X) 獲勝！"
            elif winner == -1:
                result = "Random AI (O) 獲勝！"
            else:
                result = "平局！"

        self.status_var.set(result)

        # 延遲顯示訊息框
        self.master.after(500, lambda: messagebox.showinfo("遊戲結束", result))

    def find_win_line(self):
        """找出獲勝連線"""
        board = self.env.board

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


def main():
    """主函數"""
    root = tk.Tk()
    app = TicTacToeGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
