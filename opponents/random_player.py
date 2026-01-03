"""
隨機對手
隨機選擇合法動作的簡單對手
"""
import numpy as np
from typing import List, Tuple


class RandomPlayer:
    """隨機策略玩家"""

    def __init__(self, seed: int = None):
        """
        初始化隨機玩家

        Args:
            seed: 隨機種子（可選），用於重現結果
        """
        self.rng = np.random.default_rng(seed)
        self.name = "RandomPlayer"

    def choose_action(self, legal_actions: List[Tuple[int, int]]) -> Tuple[int, int]:
        """
        從合法動作中隨機選擇一個

        Args:
            legal_actions: 合法動作列表

        Returns:
            Tuple[int, int]: 選擇的動作 (row, col)
        """
        if not legal_actions:
            raise ValueError("沒有合法動作可選擇")

        idx = self.rng.integers(0, len(legal_actions))
        return legal_actions[idx]

    def reset(self):
        """重置玩家狀態（隨機玩家不需要做什麼）"""
        pass
