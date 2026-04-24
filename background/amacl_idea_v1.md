```markdown
# AMACL 研究思路整理

> 基于多轮讨论整理，最后更新：2026-03-26

## 一、研究背景与动机

### 1.1 现有工作（WWW 2026 CRSD）的局限

已有工作 CRSD（Contrastive Reasoning Self-Distillation）被 WWW 2026 接收，核心思路是：用 LLM 生成推理理由（CoT），通过对比学习自蒸馏方式将其融入 BERT 判别式模型。

但存在根本性局限：**让 BERT "学会" LLM 的思考过程本身是个伪命题**。
- BERT（encoder-only）是单步前向传播，计算深度固定，是**System 1（直觉式模式匹配）**
- LLM 的 CoT 是多步状态转移序列，是**System 2（多步逻辑推演）**
- CRSD 实际上让 BERT 强行记住复杂文本模式，**学到的是"高阶统计相关性"，而不是"推理能力"**

### 1.2 新的研究问题

既然小模型只能做"题海战术"的模式记忆，核心痛点变成：

> **如何高效地、动态地遍历判别式模型决策边界上的盲区（Hard Cases）？**

静态离线造数据无法精准命中当前 Student 的薄弱点。需要一个能**感知 Student 当前状态、主动生成处于其"最近发展区（ZPD）"样本**的动态机制。

### 1.3 工业约束
- 线上必须使用判别式模型（BERT / Qwen-0.6B），生成式模型推理耗时不可接受
- 线上模型最大 0.6B，需要量化，输出 token 数量有限制
- 对相关性准确率要求极高（涉及真实交易场景）

## 二、方法：AMACL 框架

**AMACL**（Agentic Mutation for Automated Curriculum Learning）

### 2.1 整体架构

三个核心角色：

| 角色 | 模型 | 职责 |
|---|---|---|
| **Student** | BERT / Qwen-0.6B + 分类头 | 目标小模型，需要被持续优化 |
| **Agent（出题人）** | Qwen-7B / API 调用 | 根据 Student 当前状态，生成处于其 ZPD 的难例 |
| **Oracle（裁判）** | 冻结的强大 LLM（Qwen-Max） | 提供业务逻辑 Ground Truth，为生成样本打伪标签 |

### 2.2 Agent 的 Action Space 设计

**核心原则：不让 Agent 自由生成文本，而是调用结构化的"变异算子（Mutation Operators）"**

自由生成会导致动作空间爆炸（V^L），奖励稀疏，Agent 最终会生成乱码骗取 Reward。

#### 宏观动作：Seed 采样策略
- **A1**：高频头部 Query（防退化）
- **A2**：长尾/零展 Query（考验泛化）
- **A3**：线上高曝光低点击/低转化的 Hard Case（挖掘真实业务痛点）
- **A4**：Student 上一轮预测置信度在 0.4~0.6 之间的边界样本

#### 微观动作：变异算子库（美团 O2O 场景）

| 类别 | 算子 | 示例 |
|---|---|---|
| 实体级 | Op_Replace_Competitor（竞品替换） | 星巴克 → 瑞幸 |
| 实体级 | Op_Drop_Core_Intent（核心意图降级） | 北京烤鸭 → 全聚德（删掉"烤鸭"） |
| 实体级 | Op_Hypernym_Hyponym（上下位词替换） | 感冒药 → 医疗器械 |
| O2O属性级 | Op_Modify_Location（距离/商圈反转） | 望京烧烤 → 五道口烧烤 |
| O2O属性级 | Op_Modify_Category（类目偏移） | 理发 → 宠物美容 |
| O2O属性级 | Op_Modify_Deal（团单/供给篡改） | 双人餐 → 单人餐 |
| 语义级 | Op_Negation（逻辑取反） | 带包间的餐厅 → 不含包间 |
| 语义级 | Op_Style_Transfer（口语化/错别字改写） | 麦当劳 → 麦当佬/金拱门 |

Agent 输出标准 JSON 格式（Tool Calling 范式），由外部 Python 脚本执行字符串替换：

```json
{
  "reasoning": "Query是'苹果手机'，Doc是'Apple iPhone 15'，高度相关。保留字面相似但改变类目，制造难负例。",
  "operator": "Op_Modify_Category",
  "target_text": "Apple iPhone 15",
  "new_text": "Apple iPhone 15 硅胶保护壳"
}
```

### 2.3 Reward 设计（三层结构）

#### 第一层：R_verifiable（可验证奖励）—— 硬门控

参考 RLVR（RL with Verifiable Rewards）范式，结果可被客观验证，天然避免 Reward Hacking：

```
R_verifiable = 1  当且仅当：
  (1) JSON 格式合法且算子可执行
  (2) 执行后文本通顺（长度合理、无乱码、中文比例正常）
  (3) Oracle LLM 对 <query, doc'> 的判断与原始标签发生了反转
R_verifiable = 0  否则
```

第三条是关键：**Oracle 判断反转**是二值可验证信号，和 DeepSeek-R1 用数学答案正确性作为 Verifiable Reward 同一思路。

#### 第二层：R_learn（可学习性奖励）—— 核心创新

对应认知心理学的"最近发展区（ZPD）"概念，三种计算方式（可做消融实验）：

**方式 A（Teacher-Student Gap，推荐）**：

```
R_learn = σ( Loss_student(x', y_oracle) - Loss_oracle(x', y_oracle) )
```

Oracle 觉得简单（Loss 小）但 Student 觉得难（Loss 大）→ 高奖励。完全避免昂贵的梯度计算。

**方式 B（Ensemble Disagreement，工程最简单）**：
给 Student 挂 K 个不同初始化的分类头，计算预测方差。方差大 → Student 处于"纠结"状态 → 高奖励。

**方式 C（Margin-based）**：

```
R_learn = max(0, threshold - margin_student(x'))
```

margin = score_positive - score_negative，margin 越小说明 Student 越不确定。

#### 第三层：R_diversity（多样性奖励）—— 防模式坍塌

用信息论方式表达，惩罚算子分布偏离均匀分布：

```
R_diversity = -KL( P_current(op) || Uniform(op) )
```

#### 综合 Reward：乘法门控

```
R(x') = R_verifiable · ( α · R_learn + β · R_diversity )
```

R_verifiable 作为硬门控，质量不达标的样本整体奖励清零，Agent 无法通过牺牲质量换取 R_learn 高分。只需调 α、β 两个参数（α+β=1）。

### 2.4 协同训练 Pipeline

#### 阶段零：热身初始化
- **Student 初始化**：用线上真实点击/转化日志进行基础监督学习，建立业务常识
- **Agent 初始化**：用 Few-shot Prompting 生成算子变异示范数据，对 Agent 进行 SFT，确保输出合规 JSON

#### 核心循环（交替迭代训练）

**Step 1 — Agent 出题（冻结 Student）**
从 Seed Pool 采样 → Agent 输出 JSON → Python 执行算子变异 → 生成`<query, doc'>`

**Step 2 — Oracle 打标 + Reward 计算**
Oracle 对`<query, doc'>`重新打标（变异后标签可能已反转！）→ Student 推理得到 Loss_student → 计算三层 Reward，过滤低质量样本

**Step 3 — Agent 策略更新（GRPO）**
用收集到的轨迹（Seed → Action → Reward）通过 GRPO 更新 Agent。**第一版可跳过此步**，改用 MD 报告 + In-Context Learning 方式让 Agent 感知反馈。

**Step 4 — Student 更新（冻结 Agent）**
构建本轮错题本 →**数据混合（极重要）**：70% 基础数据 + 30% 本轮难例，防止灾难性遗忘 → Student 在混合数据上更新

#### 关键工程细节
- **标签反转问题**：Agent 变异后的样本标签必须由 Oracle 重新判定，不能默认变异后一定是负例
- **灾难性遗忘防御**：Experience Replay Buffer，正负例均衡、难易均衡
- **动态算子屏蔽**：某算子被过度使用时，用 UCB 或温度衰减强制降低采样概率

## 三、第一版实现方案（不训练 Agent，API 调用）

### Agent 感知 Reward 的 MD 报告格式

每轮迭代结束后生成报告，拼入下一轮 Agent 的 Prompt：

```markdown
## 本轮出题报告（第 N 轮）

### Student 当前状态
- 整体准确率：72.3%（上轮 68.1%，+4.2%）
- 薄弱点 Top3：竞品替换（41%）、类目偏移（55%）、口语化改写（81%）

### 本轮算子效用评估
| 算子 | 使用次数 | R_verifiable通过率 | R_learn 均值 |
|------|---------|-------------------|-------------|
| 竞品替换 | 38 | 89% | 0.73 |
| 口语化改写 | 15 | 87% | 0.09 |
| 逻辑取反 | 8 | 75% | 0.58 |

### 本轮硬约束
- 口语化改写 R_learn 极低，本轮使用次数不超过 5 次
- 逻辑取反使用不足，本轮不少于 15 次
```

第一版 Reward 主要承担**样本过滤器**角色：R_verifiable=0 直接丢弃，R_learn 低于阈值不进入 Sample Pool。

## 四、论文定位与创新点分析

### 4.1 与 SERM（字节，arXiv 2601.09515）的对比

| 维度 | SERM | AMACL（本工作） |
|---|---|---|
| 难样本来源 | **被动发现**（从线上日志筛选已有难样本） | **主动生成**（算子变异主动构造难样本） |
| 动作空间 | 无（规则筛选） | 算子化变异（有限、可控、可解释） |
| 课程学习 | 无 | 从对抗博弈中自然涌现 |
| 理论框架 | 工程 pipeline | 双层优化 + ZPD Reward |

### 4.2 核心创新点
1. **理论**：剖析 Encoder-only 模型吸收 LLM Reasoning 的物理局限（System 1 vs System 2），将问题重新定义为"动态遍历决策边界"
2. **方法**：算子化 Action Space，将无限文本生成空间降维为有限语义操作空间，天然保证质量且可解释
3. **Reward**：无需梯度计算的 ZPD Reward（Teacher-Student Loss Gap），乘法门控的三层 Reward 结构
4. **涌现现象**：课程学习从对抗博弈中自然涌现（无需人工设定难度序列）

### 4.3 目前的薄弱点

**核心问题：目前更像工程 pipeline 组合，缺乏独立的理论贡献**

审稿人可能的质疑：
- 算子化 Action Space 本质是 constrained generation，NLP 数据增强领域已有大量工作
- ZPD Reward 的 Teacher-Student Gap 在知识蒸馏文献里有类似 formulation
- 整体看是 SERM + 已有技术的组合，novelty 不足

### 4.4 加强方向（冲击 EMNLP 的路径）

**方向一（推荐）：把"课程涌现"做成核心理论贡献**

将框架形式化为双层优化问题，证明在温和假设下存在纳什均衡，且均衡点对应"Agent 恰好在 Student 能力边界上出题"。用实验验证算子分布随训练轮次的迁移曲线（从简单算子 → 复杂算子），这是论文最有价值的图表。

**方向二：深挖"判别式模型能力边界"这个科学问题**

用不同类型难例分析 Student 决策边界的形状（probing 实验 + embedding 可视化），把方法论文升级为兼具 analysis 贡献的论文。

**方向三：Oracle 打标不确定性作为更强的难例信号**

验证"多 LLM 判断分歧高的样本"是否比"Oracle 确定但 Student 不确定的样本"对 Student 提升更大。若成立，这是一个新的理论发现。

**建议路径：方向一 + 方向二组合**，形成完整研究叙事：

> WWW2026 CRSD 发现静态蒸馏的局限 → 本工作从理论上分析判别式模型能力边界 → 提出通过对抗博弈动态遍历边界的框架 → 证明课程学习从博弈中自然涌现

## 五、下一步行动

### 近期（验证可行性）
- [ ] Pilot 实验：用静态 Prompt 让 LLM 按算子生成一批数据，验证这批数据训练 BERT 是否比 CRSD 效果更好
- [ ] 验证"课程涌现"现象：跑 2~3 轮交替训练，观察 Agent 算子分布是否真的从简单向复杂迁移
- [ ] 确定 Oracle 模型选型（Qwen-Max 还是内部模型）

### 中期（完善框架）
- [ ] 实现完整的三层 Reward 计算
- [ ] 实现 Experience Replay Buffer + 数据混合策略
- [ ] 消融实验：R_learn 三种计算方式的对比

### 长期（论文冲刺）
- [ ] 理论分析：双层优化的纳什均衡证明
- [ ] 实验：判别式模型能力边界的 probing 分析
- [ ] 在公开数据集（Amazon ESCI / MS MARCO）上验证泛化性

## 六、参考文献
- **CRSD**（本组 WWW 2026）：对比推理自蒸馏，静态蒸馏基线
- **SERM**（字节，arXiv 2601.09515）：Self-Evolving Relevance Model，被动难样本挖掘
- **ATM**（EMNLP 2024，arXiv 2405.18111）：Adversarial Tuning Multi-agent，对抗训练框架
- **Adv-GRPO**（arXiv 2511.20256）：对抗奖励 + GRPO，防 Reward Hacking
- **DeepSeek-R1**：Verifiable Reward 范式
- **美团大搜**（内部，km 2742392510）：搜索体验评测 Agent，Reflexion 框架
```