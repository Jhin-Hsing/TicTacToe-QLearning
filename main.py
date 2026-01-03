"""
強化學習期末專題：井字棋遊戲策略學習（時序差分法）
主程式入口

使用 TD(0)、SARSA、Q-Learning 三種時序差分法演算法
訓練 AI agent 學習井字棋最優策略
"""
import time
import os
import sys

# 確保模組可以被正確導入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment.tictactoe import TicTacToe
from agents.td0_agent import TD0Agent
from agents.sarsa_agent import SARSAAgent
from agents.qlearning_agent import QLearningAgent
from training.train import Trainer, train_all_agents
from visualization.learning_curves import LearningCurvePlotter
from visualization.comparison_plots import ComparisonPlotter
from utils.helpers import create_output_dir, save_results_to_file, format_time


def print_header():
    """印出專題標題"""
    print("=" * 60)
    print("  強化學習期末專題：井字棋遊戲策略學習")
    print("  使用時序差分法 (Temporal Difference Learning)")
    print("=" * 60)
    print()
    print("演算法：")
    print("  1. TD(0) - 狀態值函數學習")
    print("  2. SARSA - On-policy TD 控制")
    print("  3. Q-Learning - Off-policy TD 控制")
    print()
    print("=" * 60)


def main(num_episodes=20000):
    """主程式 - 訓練模式"""
    print_header()

    # 設定超參數
    NUM_EPISODES = num_episodes      # 訓練 episode 數
    ALPHA = 0.1               # 學習率
    GAMMA = 0.99              # 折扣因子
    EPSILON = 1.0             # 初始探索率
    EPSILON_MIN = 0.01        # 最小探索率
    EPSILON_DECAY = 0.9997    # 探索率衰減

    print("\n【超參數設定】")
    print(f"  訓練 Episodes: {NUM_EPISODES}")
    print(f"  學習率 (α): {ALPHA}")
    print(f"  折扣因子 (γ): {GAMMA}")
    print(f"  初始探索率 (ε): {EPSILON}")
    print(f"  最小探索率: {EPSILON_MIN}")
    print(f"  探索率衰減: {EPSILON_DECAY}")
    print()

    # 創建輸出目錄
    output_dir = create_output_dir("output")

    # 開始計時
    start_time = time.time()

    # 訓練所有演算法
    print("\n" + "=" * 60)
    print("開始訓練...")
    print("=" * 60)

    # td0_agent, sarsa_agent, qlearning_agent, results = train_all_agents(
    #     num_episodes=NUM_EPISODES,
    #     alpha=ALPHA,
    #     gamma=GAMMA,
    #     epsilon=EPSILON,
    #     epsilon_min=EPSILON_MIN,
    #     epsilon_decay=EPSILON_DECAY
    # )
    
    # 僅訓練 TD(0)
    from training.train import train_td0_agent
    td0_agent, td0_results = train_td0_agent(
        num_episodes=NUM_EPISODES,
        alpha=ALPHA,
        gamma=GAMMA,
        epsilon=EPSILON,
        epsilon_min=EPSILON_MIN,
        epsilon_decay=EPSILON_DECAY
    )
    
    # 建構 results 字典以符合後續格式
    results = {
        'training_time': 0, # Placeholder
        'hyperparameters': {}, # Placeholder
        'TD(0)': td0_results['training'],
        'TD(0)_eval': td0_results['evaluation']
    }

    # 計算訓練時間
    training_time = time.time() - start_time
    print(f"\n總訓練時間：{format_time(training_time)}")

    # 保存結果
    results['training_time'] = format_time(training_time)
    results['hyperparameters'] = {
        'num_episodes': NUM_EPISODES,
        'alpha': ALPHA,
        'gamma': GAMMA,
        'epsilon_initial': EPSILON,
        'epsilon_min': EPSILON_MIN,
        'epsilon_decay': EPSILON_DECAY
    }
    save_results_to_file(results, 'training_results.txt', output_dir)

    # 視覺化 - 只保存不顯示
    print("\n" + "=" * 60)
    print("生成視覺化圖表（保存至 output 目錄）...")
    print("=" * 60)

    # agents = [td0_agent, sarsa_agent, qlearning_agent]
    agents = [td0_agent]

    # 1. 繪製各個 Agent 的學習曲線
    curve_plotter = LearningCurvePlotter()

    print("\n保存 TD(0) 學習曲線...")
    curve_plotter.plot_all_metrics(
        td0_agent,
        save_path=os.path.join(output_dir, 'td0_learning_curves.png'),
        show=False
    )

    # print("保存 SARSA 學習曲線...")
    # curve_plotter.plot_all_metrics(
    #     sarsa_agent,
    #     save_path=os.path.join(output_dir, 'sarsa_learning_curves.png'),
    #     show=False
    # )

    # print("保存 Q-Learning 學習曲線...")
    # curve_plotter.plot_all_metrics(
    #     qlearning_agent,
    #     save_path=os.path.join(output_dir, 'qlearning_learning_curves.png'),
    #     show=False
    # )

    # 2. 繪製演算法比較圖
    # comparison_plotter = ComparisonPlotter()

    # # 準備評估結果
    # eval_results = {
    #     'TD(0)': results['TD(0)_eval'],
    #     # 'SARSA': results['SARSA_eval'],
    #     # 'Q-Learning': results['Q-Learning_eval']
    # }

    # print("\n保存演算法綜合比較圖...")
    # comparison_plotter.plot_comprehensive_comparison(
    #     agents,
    #     eval_results,
    #     save_path=os.path.join(output_dir, 'algorithm_comparison.png'),
    #     show=False
    # )

    # print("保存勝率比較圖...")
    # comparison_plotter.compare_win_rates(
    #     agents,
    #     save_path=os.path.join(output_dir, 'win_rate_comparison.png'),
    #     show=False
    # )

    # print("保存收斂速度比較圖...")
    # comparison_plotter.compare_convergence(
    #     agents,
    #     threshold=0.7,
    #     save_path=os.path.join(output_dir, 'convergence_comparison.png'),
    #     show=False
    # )

    # 3. 保存訓練好的模型
    print("\n" + "=" * 60)
    print("保存訓練模型...")
    print("=" * 60)

    models_dir = create_output_dir("models")
    td0_agent.save(os.path.join(models_dir, 'td0_agent.pkl'))
    # sarsa_agent.save(os.path.join(models_dir, 'sarsa_agent.pkl'))
    # qlearning_agent.save(os.path.join(models_dir, 'qlearning_agent.pkl'))
    print("模型已保存至 models 目錄")

    # 最終結果總結
    print("\n" + "=" * 60)
    print("訓練完成！結果總結")
    print("=" * 60)

    print("\n【最終評估結果】（對戰 1000 場隨機對手）")
    # for name in ['TD(0)', 'SARSA', 'Q-Learning']:
    for name in ['TD(0)']:
        eval_result = results[f'{name}_eval']
        print(f"\n{name}:")
        print(f"  勝率：{eval_result['win_rate']*100:.1f}%")
        print(f"  敗率：{eval_result['loss_rate']*100:.1f}%")
        print(f"  平局率：{eval_result['draw_rate']*100:.1f}%")

    print("\n【輸出檔案】")
    print(f"  訓練結果：{os.path.join(output_dir, 'training_results.txt')}")
    print(f"  學習曲線：{output_dir}/")
    print(f"  比較圖表：{output_dir}/")
    print(f"  訓練模型：{models_dir}/")

    print("\n" + "=" * 60)
    print("專題執行完成！")
    print("=" * 60)

    print("\n提示：執行以下指令開啟遊戲介面：")
    print("  python play_game.py")


def gui_mode():
    """啟動圖形介面"""
    try:
        from play_game import main as play_main
        play_main()
    except ImportError as e:
        print(f"無法啟動圖形介面: {e}")
        print("請確保 tkinter 已正確安裝")


def interactive_mode():
    """互動模式：人類玩家對戰 AI（終端機版）"""
    print("\n【互動模式：人類 vs AI】")
    print("你是 O，AI 是 X")
    print("輸入格式：row,col (例如：1,1 表示中間格子)")
    print("輸入 'q' 退出\n")

    env = TicTacToe()

    # 嘗試載入模型
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    qlearning_path = os.path.join(models_dir, 'qlearning_agent.pkl')

    if os.path.exists(qlearning_path):
        print("載入已訓練的 Q-Learning 模型...")
        agent = QLearningAgent()
        agent.load(qlearning_path)
        agent.epsilon = 0
    else:
        print("找不到模型，使用未訓練的 Agent（較弱）...")
        agent = QLearningAgent(epsilon=0)

    state = env.reset()
    env.render()

    while not env.done:
        if env.current_player == 1:  # AI 的回合
            print("AI 思考中...")
            legal_actions = env.get_legal_actions()
            action = agent.choose_action(state, legal_actions)
            print(f"AI 下在：{action}")
            state, _, done, info = env.step(action)
        else:  # 人類回合
            while True:
                try:
                    user_input = input("你的回合，輸入位置 (row,col): ").strip()
                    if user_input.lower() == 'q':
                        print("遊戲結束")
                        return

                    parts = user_input.split(',')
                    row, col = int(parts[0]), int(parts[1])

                    if (row, col) not in env.get_legal_actions():
                        print("無效位置，請重試")
                        continue

                    state, _, done, info = env.step((row, col))
                    break
                except (ValueError, IndexError):
                    print("輸入格式錯誤，請使用 row,col 格式（例如：1,1）")

        env.render()

    # 顯示結果
    if env.winner == 1:
        print("AI 獲勝！")
    elif env.winner == -1:
        print("恭喜！你贏了！")
    else:
        print("平局！")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='井字棋強化學習專題')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'play', 'gui'],
                       help='執行模式：train=訓練, play=終端機對戰, gui=圖形介面')
    parser.add_argument('--episodes', type=int, default=20000,
                       help='訓練 episode 數量（預設：20000）')

    args = parser.parse_args()

    if args.mode == 'train':
        main(num_episodes=args.episodes)
    elif args.mode == 'play':
        interactive_mode()
    elif args.mode == 'gui':
        gui_mode()
