"""
訓練系統
使用 TD(0) 時序差分法訓練井字棋 AI
"""
import numpy as np
from typing import Dict, List, Tuple, Any
from tqdm import tqdm
import sys
import os

# 添加專案根目錄到路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.tictactoe import TicTacToe
from agents.td0_agent import TD0Agent
from agents.sarsa_agent import SARSAAgent
from agents.qlearning_agent import QLearningAgent
from opponents.random_player import RandomPlayer


class Trainer:
    """強化學習訓練器"""

    def __init__(self, env: TicTacToe = None):
        """
        初始化訓練器

        Args:
            env: 井字棋環境
        """
        self.env = env if env else TicTacToe()

    def train(
        self,
        agent: Any,
        opponent: RandomPlayer,
        num_episodes: int = 50000,
        verbose: bool = True
    ) -> Dict:
        """
        訓練 Agent

        Args:
            agent: Agent (TD0, SARSA, or Q-Learning)
            opponent: 對手（隨機策略）
            num_episodes: 訓練回合數
            verbose: 是否顯示進度

        Returns:
            訓練結果統計
        """
        wins, losses, draws = 0, 0, 0

        iterator = tqdm(range(num_episodes), desc=f"訓練 {agent.name}") if verbose else range(num_episodes)

        for episode in iterator:
            state = self.env.reset()
            agent.reset()
            opponent.reset()

            total_reward = 0
            step_count = 0
            done = False
            
            # For SARSA: we need to choose the first action
            action = None
            if agent.name == 'SARSA':
                legal_actions = self.env.get_legal_actions()
                action = agent.choose_action(state, legal_actions)

            while not done:
                if self.env.current_player == 1:  # AI (X) 的回合
                    legal_actions = self.env.get_legal_actions()
                    
                    # Choose action if not already chosen (non-SARSA or first step)
                    if action is None:
                        action = agent.choose_action(state, legal_actions)

                    # 執行動作
                    next_state, reward, done, info = self.env.step(action)
                    step_count += 1

                    if not done:
                        # 對手回合
                        opp_legal = self.env.get_legal_actions()
                        if opp_legal:
                            opp_action = opponent.choose_action(opp_legal)
                            next_state, _, done, info = self.env.step(opp_action)

                            # 如果對手贏了，AI 得到負獎勵
                            if done and info['winner'] == -1:
                                reward = -1.0

                    # 更新 Agent
                    next_legal = self.env.get_legal_actions() if not done else []
                    
                    if agent.name == 'SARSA':
                        next_action = None
                        if not done:
                            next_action = agent.choose_action(next_state, next_legal)
                        
                        agent.update(state, action, reward, next_state, next_action, done)
                        
                        # Prepare for next iteration
                        action = next_action
                    else:
                        # TD(0) and Q-Learning
                        agent.update(state, action, reward, next_state, next_legal, done)
                        action = None # Reset action so it's chosen next loop

                    total_reward += reward
                    state = next_state

                else:  # 對手先手（通常不會發生，但以防萬一）
                    legal_actions = self.env.get_legal_actions()
                    opp_action = opponent.choose_action(legal_actions)
                    state, _, done, info = self.env.step(opp_action)
                    # Note: If opponent plays first, we just loop back to AI turn
                    # But for SARSA, if AI hasn't acted yet, we don't have an 'action' to update?
                    # Since AI is player 1 (X) and always goes first in reset(), this block might be for robustness.
                    # If we are in this block, state changes.
                    # For SARSA, if we had a pending action/state, this would disrupt it. 
                    # But assuming X goes first, this else block only runs if O goes first (which reset() defaults to X).

            # 記錄結果
            agent.record_episode(total_reward, step_count)
            agent.decay_epsilon()

            if info['winner'] == 1:
                wins += 1
            elif info['winner'] == -1:
                losses += 1
            else:
                draws += 1

            # 更新進度條
            if verbose and (episode + 1) % 1000 == 0:
                recent_wins = sum(1 for r in agent.episode_rewards[-1000:] if r > 0)
                win_rate = recent_wins / 10  # 最近 1000 場的勝率
                iterator.set_postfix({
                    '近期勝率': f"{win_rate:.1f}%",
                    'ε': f"{agent.epsilon:.3f}",
                    '總勝': wins,
                    '總敗': losses
                })

        return {
            'wins': wins,
            'losses': losses,
            'draws': draws,
            'win_rate': wins / num_episodes,
            'loss_rate': losses / num_episodes,
            'draw_rate': draws / num_episodes
        }

    def evaluate(
        self,
        agent: Any,
        opponent: RandomPlayer,
        num_games: int = 1000
    ) -> Dict:
        """
        評估 Agent 的表現（不探索）

        Args:
            agent: 要評估的 Agent
            opponent: 對手
            num_games: 評估遊戲數量

        Returns:
            評估結果
        """
        wins, losses, draws = 0, 0, 0
        original_epsilon = agent.epsilon
        agent.epsilon = 0  # 評估時不探索

        for _ in range(num_games):
            state = self.env.reset()
            done = False

            while not done:
                if self.env.current_player == 1:  # AI 的回合
                    legal_actions = self.env.get_legal_actions()
                    action = agent.choose_action(state, legal_actions)
                    state, _, done, info = self.env.step(action)
                else:  # 對手回合
                    legal_actions = self.env.get_legal_actions()
                    action = opponent.choose_action(legal_actions)
                    state, _, done, info = self.env.step(action)

            if info['winner'] == 1:
                wins += 1
            elif info['winner'] == -1:
                losses += 1
            else:
                draws += 1

        agent.epsilon = original_epsilon  # 恢復探索率

        return {
            'wins': wins,
            'losses': losses,
            'draws': draws,
            'win_rate': wins / num_games,
            'loss_rate': losses / num_games,
            'draw_rate': draws / num_games
        }

    def train_self_play(
        self,
        agent: Any,
        num_episodes: int = 50000,
        verbose: bool = True
    ) -> Dict:
        """
        自我對弈訓練 (Self-Play)
        Agent 擔任 X，同時也擔任 O 的策略來源，雙方都從經驗中學習
        """
        wins, losses, draws = 0, 0, 0
        iterator = tqdm(range(num_episodes), desc=f"自我對弈 {agent.name}") if verbose else range(num_episodes)

        for episode in iterator:
            state = self.env.reset()
            agent.reset()
            
            # 記錄 X 和 O 的狀態序列
            x_history = [] # (s, a)
            o_history = [] # (s, a)
            
            step_count = 0
            done = False
            while not done:
                # 當前玩家選擇動作
                legal_actions = self.env.get_legal_actions()
                action = agent.choose_action(state, legal_actions)
                step_count += 1
                
                # 記錄歷史
                current_player = self.env.current_player
                if current_player == 1:
                    x_history.append((state.copy(), action))
                else:
                    o_history.append((state.copy(), action))
                
                # 執行
                next_state, reward, done, info = self.env.step(action)
                
                if done:
                    # 遊戲結束，更新雙方 Q 值
                    winner = info['winner']
                    
                    # 更新 X 的最後一步
                    if x_history:
                        s_x, a_x = x_history[-1]
                        r_x = 1.0 if winner == 1 else (-1.0 if winner == -1 else 0.0)
                        agent.update(s_x, a_x, r_x, next_state, [], True)
                    
                    # 更新 O 的最後一步
                    if o_history:
                        s_o, a_o = o_history[-1]
                        r_o = 1.0 if winner == -1 else (-1.0 if winner == 1 else 0.0)
                        agent.update(s_o, a_o, r_o, next_state, [], True)
                        
                    # 記錄勝負
                    if winner == 1: wins += 1
                    elif winner == -1: losses += 1
                    else: draws += 1
                    
                    # 記錄統計數據 (以 X 的視角記錄)
                    # 如果 X 贏 = 1, O 贏 = -1, 平局 = 0
                    episode_reward = 1.0 if winner == 1 else (-1.0 if winner == -1 else 0.0)
                    agent.record_episode(episode_reward, step_count)
                
                # 注意：這裡不進行中間步的更新，或者您可以實作中間步更新邏輯
                # 為了簡單起見，我們在 done 時處理。
                # 但更標準的做法是每步更新：
                elif current_player == -1 and x_history:
                    # 如果 O 剛下完，我們可以更新 X 之前的動作（因為 X 觀察到了 O 的回應）
                    s_prev_x, a_prev_x = x_history[-1]
                    # 這裡 reward 為 0 因為遊戲還沒結束
                    agent.update(s_prev_x, a_prev_x, 0, next_state, self.env.get_legal_actions(), False)
                elif current_player == 1 and o_history:
                    # 如果 X 剛下完，更新 O 之前的動作
                    s_prev_o, a_prev_o = o_history[-1]
                    agent.update(s_prev_o, a_prev_o, 0, next_state, self.env.get_legal_actions(), False)

                state = next_state

            agent.decay_epsilon()

        return {'wins': wins, 'losses': losses, 'draws': draws}


def train_agent(
    agent_class,
    num_episodes: int = 50000,
    alpha: float = 0.3,
    gamma: float = 0.95,
    epsilon: float = 1.0,
    epsilon_min: float = 0.01,
    epsilon_decay: float = 0.9999
) -> Tuple[Any, Dict]:
    """
    更新後的訓練流程：先隨機對弈，後自我對弈
    """
    env = TicTacToe()
    trainer = Trainer(env)
    
    agent = agent_class(
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay
    )

    # 階段 1：隨機對弈 (20%) - 學習基礎規則
    print(f"\n[階段 1] {agent.name} 隨機對弈訓練...")
    trainer.train(agent, RandomPlayer(), int(num_episodes * 0.2))

    # 階段 2：自我對弈 (80%) - 學習高階防守與佈局
    print(f"\n[階段 2] {agent.name} 自我對弈訓練...")
    # 重置探索率，確保在自我對弈時能探索新的防守策略
    agent.epsilon = 0.6
    trainer.train_self_play(agent, int(num_episodes * 0.8))

    # 評估
    print("\n" + "=" * 60)
    print("評估訓練結果 (vs 隨機對手)...")
    eval_results = trainer.evaluate(agent, RandomPlayer(), num_games=1000)

    print(f"\n最終勝率：{eval_results['win_rate']*100:.1f}%")

    return agent, {'training': eval_results, 'evaluation': eval_results}


# 保持向後兼容的介面，但擴充功能
def train_all_agents(
    num_episodes: int = 50000,
    alpha: float = 0.3,
    gamma: float = 0.95,
    epsilon: float = 1.0,
    epsilon_min: float = 0.01,
    epsilon_decay: float = 0.9999
):
    """訓練所有 Agent"""
    
    # 訓練 TD(0)
    td0_agent, td0_results = train_agent(
        TD0Agent, num_episodes, alpha, gamma, epsilon, epsilon_min, epsilon_decay
    )
    
    # 訓練 SARSA
    sarsa_agent, sarsa_results = train_agent(
        SARSAAgent, num_episodes, alpha, gamma, epsilon, epsilon_min, epsilon_decay
    )
    
    # 訓練 Q-Learning
    qlearning_agent, qlearning_results = train_agent(
        QLearningAgent, num_episodes, alpha, gamma, epsilon, epsilon_min, epsilon_decay
    )

    all_results = {
        'TD(0)': td0_results['training'],
        'TD(0)_eval': td0_results['evaluation'],
        'SARSA': sarsa_results['training'],
        'SARSA_eval': sarsa_results['evaluation'],
        'Q-Learning': qlearning_results['training'],
        'Q-Learning_eval': qlearning_results['evaluation']
    }

    return td0_agent, sarsa_agent, qlearning_agent, all_results


def train_td0_agent(num_episodes=50000, **kwargs):
    """保留舊函數名稱以防萬一"""
    return train_agent(TD0Agent, num_episodes=num_episodes, **kwargs)


if __name__ == "__main__":
    agent, results = train_td0_agent(num_episodes=50000)

    print("\n" + "=" * 60)
    print("訓練完成！")
    print(f"學習到的狀態-動作對數量：{len(agent.Q)}")
    print("=" * 60)
