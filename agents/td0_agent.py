"""
TD(0) Agent
使用時序差分法 TD(0) 學習動作值函數 Q(s, a)
針對井字棋遊戲優化
"""
import numpy as np
from typing import List, Tuple, Dict
from collections import defaultdict
import pickle


class TD0Agent:
    """
    TD(0) Agent - 使用時序差分法學習

    核心公式：
    Q(s, a) ← Q(s, a) + α * [r + γ * max_a' Q(s', a') - Q(s, a)]

    這是 TD(0) 控制方法，結合了時序差分學習和 Q 值更新
    """

    def __init__(
        self,
        alpha: float = 0.3,          # 學習率（較高以加速學習）
        gamma: float = 0.95,          # 折扣因子
        epsilon: float = 1.0,         # 初始探索率
        epsilon_min: float = 0.01,    # 最小探索率
        epsilon_decay: float = 0.9995 # 探索率衰減
    ):
        """
        初始化 TD(0) Agent

        Args:
            alpha: 學習率 - 控制學習速度
            gamma: 折扣因子 - 未來獎勵的權重
            epsilon: 探索率 - ε-greedy 策略中的探索機率
            epsilon_min: 最小探索率
            epsilon_decay: 每個 episode 後探索率的衰減係數
        """
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Q-table: Q(s, a) - 狀態-動作值函數
        # 使用 (state_key, action) 作為鍵
        self.Q: Dict[Tuple[str, Tuple[int, int]], float] = defaultdict(float)

        self.name = "TD(0)"

        # 記錄學習歷史
        self.episode_rewards = []
        self.episode_lengths = []
        self.epsilon_history = []

    def get_state_key(self, board: np.ndarray) -> str:
        """
        將棋盤狀態轉換為字串鍵值

        Args:
            board: 3x3 棋盤狀態矩陣

        Returns:
            棋盤狀態的字串表示
        """
        return str(board.flatten().tolist())

    def get_q_value(self, state: np.ndarray, action: Tuple[int, int]) -> float:
        """獲取 Q(s, a) 值"""
        key = (self.get_state_key(state), action)
        return self.Q[key]

    def get_max_q_value(self, state: np.ndarray, legal_actions: List[Tuple[int, int]]) -> float:
        """獲取狀態 s 的最大 Q 值: max_a Q(s, a)"""
        if not legal_actions:
            return 0.0

        state_key = self.get_state_key(state)
        return max(self.Q[(state_key, a)] for a in legal_actions)

    def choose_action(
        self,
        state: np.ndarray,
        legal_actions: List[Tuple[int, int]],
        env_clone_func=None  # 保留參數以兼容舊介面
    ) -> Tuple[int, int]:
        """
        使用 ε-greedy 策略選擇動作

        ε-greedy 策略：
        - 以機率 ε 隨機選擇動作（探索）
        - 以機率 1-ε 選擇 Q 值最高的動作（利用）

        Args:
            state: 當前棋盤狀態
            legal_actions: 合法動作列表

        Returns:
            選擇的動作 (row, col)
        """
        if not legal_actions:
            raise ValueError("沒有合法動作可選擇")

        # 探索：以機率 ε 隨機選擇
        if np.random.random() < self.epsilon:
            idx = np.random.randint(len(legal_actions))
            return legal_actions[idx]

        # 利用：選擇 Q 值最高的動作
        state_key = self.get_state_key(state)

        best_action = legal_actions[0]
        best_q = self.Q[(state_key, legal_actions[0])]

        for action in legal_actions[1:]:
            q = self.Q[(state_key, action)]
            if q > best_q:
                best_q = q
                best_action = action

        return best_action

    def update(
        self,
        state: np.ndarray,
        action: Tuple[int, int],
        reward: float,
        next_state: np.ndarray,
        next_legal_actions: List[Tuple[int, int]],
        done: bool
    ):
        """
        TD(0) 更新規則

        公式：Q(s,a) ← Q(s,a) + α * [r + γ * max_a' Q(s',a') - Q(s,a)]

        其中：
        - α: 學習率
        - r: 即時獎勵
        - γ: 折扣因子
        - max_a' Q(s',a'): 下一狀態的最大 Q 值

        Args:
            state: 當前狀態 s
            action: 執行的動作 a
            reward: 獲得的獎勵 r
            next_state: 下一個狀態 s'
            next_legal_actions: 下一個狀態的合法動作列表
            done: 遊戲是否結束
        """
        state_key = self.get_state_key(state)
        current_q = self.Q[(state_key, action)]

        if done:
            # 終止狀態：TD target = r
            td_target = reward
        else:
            # 非終止狀態：TD target = r + γ * max_a' Q(s', a')
            max_next_q = self.get_max_q_value(next_state, next_legal_actions)
            td_target = reward + self.gamma * max_next_q

        # TD 誤差：δ = TD_target - Q(s, a)
        td_error = td_target - current_q

        # 更新 Q 值：Q(s, a) ← Q(s, a) + α * δ
        self.Q[(state_key, action)] += self.alpha * td_error

    def decay_epsilon(self):
        """衰減探索率"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def record_episode(self, total_reward: float, length: int):
        """記錄 episode 數據"""
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(length)
        self.epsilon_history.append(self.epsilon)

    def reset(self):
        """重置 episode 狀態（不清除學習到的 Q 值）"""
        pass

    def save(self, filepath: str):
        """保存模型到檔案"""
        # 將 Q-table 轉換為可序列化格式
        q_serializable = {}
        for (state_key, action), value in self.Q.items():
            key_str = f"{state_key}|{action}"
            q_serializable[key_str] = value

        data = {
            'Q': q_serializable,
            'alpha': self.alpha,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'epsilon_min': self.epsilon_min,
            'epsilon_decay': self.epsilon_decay,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'epsilon_history': self.epsilon_history
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    def load(self, filepath: str):
        """從檔案載入模型"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        # 還原 Q-table
        self.Q = defaultdict(float)
        for key_str, value in data['Q'].items():
            parts = key_str.split('|')
            state_key = parts[0]
            action = eval(parts[1])  # 將字串轉回 tuple
            self.Q[(state_key, action)] = value

        self.alpha = data['alpha']
        self.gamma = data['gamma']
        self.epsilon = data['epsilon']
        self.epsilon_min = data['epsilon_min']
        self.epsilon_decay = data['epsilon_decay']
        self.episode_rewards = data.get('episode_rewards', [])
        self.episode_lengths = data.get('episode_lengths', [])
        self.epsilon_history = data.get('epsilon_history', [])

    def get_stats(self) -> dict:
        """獲取統計資訊"""
        return {
            'name': self.name,
            'num_state_action_pairs': len(self.Q),
            'epsilon': self.epsilon,
            'total_episodes': len(self.episode_rewards)
        }
