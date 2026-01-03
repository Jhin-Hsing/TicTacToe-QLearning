"""
井字棋遊戲環境
實作標準的 3x3 井字棋遊戲邏輯
"""
import numpy as np
from typing import Tuple, List, Optional


class TicTacToe:
    """井字棋環境類別"""

    def __init__(self):
        """初始化遊戲環境"""
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1  # 1 代表 X (AI), -1 代表 O (對手)
        self.done = False
        self.winner = None

    def reset(self) -> np.ndarray:
        """
        重置遊戲到初始狀態

        Returns:
            np.ndarray: 初始棋盤狀態
        """
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1
        self.done = False
        self.winner = None
        return self.board.copy()

    def get_legal_actions(self) -> List[Tuple[int, int]]:
        """
        獲取所有合法動作（空格位置）

        Returns:
            List[Tuple[int, int]]: 合法動作列表，每個動作是 (row, col) 座標
        """
        legal_actions = []
        for i in range(3):
            for j in range(3):
                if self.board[i, j] == 0:
                    legal_actions.append((i, j))
        return legal_actions

    def step(self, action: Tuple[int, int]) -> Tuple[np.ndarray, float, bool, dict]:
        """
        執行一個動作

        Args:
            action: (row, col) 座標

        Returns:
            state: 新的棋盤狀態
            reward: 獎勵值
            done: 遊戲是否結束
            info: 額外資訊
        """
        if self.done:
            raise ValueError("遊戲已經結束，請先 reset()")

        row, col = action

        # 檢查動作是否合法
        if self.board[row, col] != 0:
            raise ValueError(f"位置 ({row}, {col}) 已經有棋子了")

        # 執行動作
        self.board[row, col] = self.current_player

        # 檢查遊戲是否結束
        winner = self._check_winner()

        if winner is not None:
            self.done = True
            self.winner = winner
            # 從當前玩家的角度給予獎勵
            if winner == self.current_player:
                reward = 1.0  # 贏了
            elif winner == -self.current_player:
                reward = -1.0  # 輸了
            else:
                reward = 0.0  # 平局
        elif len(self.get_legal_actions()) == 0:
            # 棋盤滿了但沒有贏家，平局
            self.done = True
            self.winner = 0
            reward = 0.0
        else:
            # 遊戲繼續
            reward = 0.0
            # 切換玩家
            self.current_player = -self.current_player

        info = {
            'winner': self.winner,
            'current_player': self.current_player
        }

        return self.board.copy(), reward, self.done, info

    def _check_winner(self) -> Optional[int]:
        """
        檢查是否有贏家

        Returns:
            1: X 贏了
            -1: O 贏了
            0: 平局（棋盤滿了）
            None: 遊戲繼續
        """
        # 檢查行
        for i in range(3):
            if abs(self.board[i, :].sum()) == 3:
                return self.board[i, 0]

        # 檢查列
        for j in range(3):
            if abs(self.board[:, j].sum()) == 3:
                return self.board[0, j]

        # 檢查對角線
        if abs(self.board.diagonal().sum()) == 3:
            return self.board[1, 1]

        # 檢查反對角線
        if abs(np.fliplr(self.board).diagonal().sum()) == 3:
            return self.board[1, 1]

        return None

    def is_terminal(self) -> bool:
        """檢查遊戲是否結束"""
        return self.done

    def get_state_key(self) -> str:
        """
        將棋盤狀態轉換為字串鍵值，用於 Q-table 索引

        Returns:
            str: 棋盤狀態的字串表示
        """
        return str(self.board.flatten().tolist())

    def render(self, mode='human'):
        """
        顯示棋盤

        Args:
            mode: 顯示模式，預設為 'human'
        """
        symbols = {0: '.', 1: 'X', -1: 'O'}
        print("\n當前棋盤:")
        print("  0 1 2")
        for i in range(3):
            print(f"{i} ", end="")
            for j in range(3):
                print(symbols[self.board[i, j]], end=" ")
            print()
        print()

    def clone(self):
        """
        複製當前環境狀態

        Returns:
            TicTacToe: 新的環境副本
        """
        new_env = TicTacToe()
        new_env.board = self.board.copy()
        new_env.current_player = self.current_player
        new_env.done = self.done
        new_env.winner = self.winner
        return new_env

    def get_symmetries(self) -> List[np.ndarray]:
        """
        獲取棋盤的所有對稱狀態（用於提高學習效率）
        包含旋轉和鏡像

        Returns:
            List[np.ndarray]: 所有對稱狀態列表
        """
        symmetries = []
        board = self.board.copy()

        # 4 種旋轉
        for k in range(4):
            symmetries.append(np.rot90(board, k))

        # 鏡像
        symmetries.append(np.fliplr(board))
        symmetries.append(np.flipud(board))

        # 對角線翻轉
        symmetries.append(board.T)
        symmetries.append(np.fliplr(board).T)

        return symmetries
