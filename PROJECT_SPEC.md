# 強化學習期末專題規劃書：基於時序差分法的井字棋策略研究

## 1. 專題目的 (Project Objective)

本專題旨在探討並實作**強化學習 (Reinforcement Learning, RL)** 中的**時序差分學習 (Temporal Difference Learning, TD Learning)** 方法，應用於解決井字棋 (Tic-Tac-Toe) 賽局問題。

主要目標如下：
1.  **環境建置**：實作符合 OpenAI Gym 介面規範的井字棋環境，定義清晰的狀態轉移邏輯。
2.  **演算法比較**：實作並比較三種核心 TD 演算法的效能與收斂特性：
    *   **TD(0)** (狀態價值函數估計)
    *   **SARSA** (On-policy 動作價值控制)
    *   **Q-Learning** (Off-policy 動作價值控制)
3.  **策略分析**：透過勝率曲線、收斂速度與策略視覺化，分析不同演算法在離散零和賽局中的表現差異。

---

## 2. 強化學習環境定義 (RL Environment Definition)

本問題被建模為一個**馬可夫決策過程 (Markov Decision Process, MDP)**，由五個核心元素組成 $(S, A, P, R, \gamma)$：

### 2.1 環境 (Environment)
*   **定義**：標準 3x3 井字棋盤。
*   **規則**：雙人回合制零和遊戲。先連成一線（橫、直、斜）者獲勝。
*   **對手設定**：訓練階段主要對抗隨機策略 (Random Agent) 或自我對弈，測試階段評估對抗隨機策略的勝率。

### 2.2 智能體 (Agent)
*   **定義**：學習者 (Learner)，在本專題中執棋方為 'X' (先手或後手皆可訓練，預設為先手)。
*   **目標**：最大化累積獎勵 (Cumulative Reward)，即最大化獲勝機率。

### 2.3 狀態 (State, $S$)
*   **定義**：棋盤當下的盤面配置。
*   **表示法**：使用長度為 9 的向量或 3x3 矩陣展平後的字串作為特徵 (Hashable Key)。
    *   $s \in \{0, 1, -1\}^9$
    *   0: 空格, 1: 我方 (Agent), -1: 對手
*   **狀態空間大小**：上限為 $3^9 = 19683$，但在井字棋規則限制下，合法狀態數約為 5,478 個，適合使用表格法 (Tabular Method)。

### 2.4 動作 (Action, $A$)
*   **定義**：在棋盤上的合法落子位置。
*   **動作空間**：$a \in \{(i, j) | 0 \le i, j \le 2\}$。
*   **合法性**：僅能選擇當前狀態 $s$ 中值為 0 (空格) 的位置。若棋盤已滿，則無動作可執行。

### 2.5 獎勵 (Reward, $R$)
獎勵函數設計為稀疏獎勵 (Sparse Reward)，僅在遊戲結束時給予反饋：
*   $r = +1$：**獲勝 (Win)**。
*   $r = -1$：**落敗 (Loss)**。
*   $r = 0$：**平局 (Draw)** 或 **遊戲進行中 (Intermediate step)**。
*   **目的**：引導 Agent 追求勝利並避免失敗，同時不排斥平局（相較於輸棋）。

---

## 3. 使用之強化學習演算法 (Algorithms)

本專題採用表格法 (Tabular Methods) 實作以下三種演算法：

### 3.1 TD(0) - 狀態價值學習 (State Value Learning)
*   **核心概念**：學習狀態價值函數 $V(s)$，代表在狀態 $s$ 下預期能獲得的總回報。
*   **更新公式**：
    $$V(s_t) \leftarrow V(s_t) + \alpha [R_{t+1} + \gamma V(s_{t+1}) - V(s_t)]$$
*   **決策方式**：由於僅學習 $V(s)$，決策時需透過環境模型預測下一步狀態 $s'$，並選擇 $\arg\max_a V(s')$。

### 3.2 SARSA - On-policy 控制 (State-Action-Reward-State-Action)
*   **核心概念**：學習動作價值函數 $Q(s, a)$。更新時使用「實際執行的下一個動作 $a'$」。
*   **更新公式**：
    $$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha [R_{t+1} + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)]$$
*   **特性**：On-policy，學習到的策略會考慮到探索過程 (Exploration) 的風險，通常較為保守。

### 3.3 Q-Learning - Off-policy 控制
*   **核心概念**：學習動作價值函數 $Q(s, a)$。更新時假設下一步採取「最佳動作」，與實際是否執行無關。
*   **更新公式**：
    $$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha [R_{t+1} + \gamma \max_{a} Q(s_{t+1}, a) - Q(s_t, a_t)]$$
*   **特性**：Off-policy，目標是學習全域最佳策略 (Optimal Policy)。

### 3.4 策略選擇 (Action Selection)
*   採用 **$\epsilon$-greedy** 策略平衡探索 (Exploration) 與利用 (Exploitation)。
*   隨著訓練回合數 (Episodes) 增加，$\epsilon$ 值呈指數衰減，初期多探索，後期多利用。

---

## 4. 實作與運用規劃 (Implementation Plan)

### 4.1 系統架構
1.  **`Environment`**: 封裝遊戲邏輯，提供 `reset()` 和 `step(action)` 方法。
2.  **`Agent`**: 封裝 Q-table/V-table 與演算法更新邏輯 (`update()`, `choose_action()`)。
3.  **`Training Loop`**: 控制訓練流程，負責 Agent 與 Environment 的互動，並記錄數據。
4.  **`Visualization`**: 繪製 Reward 曲線、勝率變化圖。

### 4.2 訓練流程 (Training Pipeline)
1.  初始化 Agent (Q-table 全 0) 與 Environment。
2.  進行 $N$ 個 Episodes 的訓練（例如：20,000 場）。
3.  每一場遊戲中：
    *   Agent 觀察狀態 $S$。
    *   Agent 根據 $\epsilon$-greedy 選擇動作 $A$。
    *   Environment 執行 $A$，回傳獎勵 $R$ 與新狀態 $S'$ (包含對手行動後的結果)。
    *   Agent 根據 ($S, A, R, S'$) 更新價值函數。
4.  定期評估：每 $M$ 場訓練後，關閉探索 ($\epsilon=0$) 進行測試，記錄勝率。

### 4.3 預期成果
*   產出各演算法的學習曲線 (Learning Curves)。
*   產出各演算法在相同訓練量下的勝率比較表。
*   驗證 Q-Learning 在井字棋這類確定性賽局中能收斂至最佳策略（即面對隨機對手勝率最高，面對完美對手不輸）。
