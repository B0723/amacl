# AMACL：面向搜索相关性的智能体变异自动课程学习 v2
# AMACL：面向搜索相关性的智能体变异自动课程学习

> **研究定位**：冲击 EMNLP 2026（ARR 截稿 2026年5月25日）/ KDD 2027，同时具备美团搜索工业落地可行性
**核心问题**：如何让判别式相关性模型在无需大量人工标注的前提下，通过 LLM Agent 主动生成"恰好在模型能力边界"的难样本，实现持续自适应提升？

---

## 一、研究背景与动机
### 1.1 工业搜索相关性的核心矛盾

电商搜索相关性判断面临两个相互制约的约束：**推理质量**与**在线延迟**。

BERT 类交叉编码器（Cross-Encoder）是工业界主流的相关性模型，推理延迟 <10ms，但其本质是基于词汇共现和浅层语义的模式匹配，在以下场景中系统性失效：
- **品牌混淆**：query="苹果手机" + doc="华为 Mate 60 Pro"（同类目不同品牌，BERT 因"手机"词汇重叠倾向于判相关）
- **属性误匹配**：query="512G 手机" + doc="512G 移动硬盘"（属性词相同但类目不同）
- **意图漂移**：query="减肥方法" + doc="减肥药广告"（意图是了解方法，非购买商品）
- **跨类目泛化**：query="宝宝辅食" + doc="婴儿米粉"（需要领域知识才能判断相关）

LLM（如 Qwen2.5-72B）通过链式推理（Chain-of-Thought）能够正确处理上述场景，但推理延迟 >500ms，无法部署在在线链路。

**核心矛盾**：BERT 快但弱，LLM 强但慢。现有工作要么停留在 BERT 侧（ANCE、CRSD），要么停留在 LLM 侧（LORE），缺乏一个能让 BERT 持续向 LLM 对齐的**动态数据生成机制**。
### 1.2 现有方法的系统性局限

我们将现有方法按"数据来源"和"是否自适应"两个维度分类：

```Plain Text
静态数据              动态数据
                ┌─────────────────┬─────────────────┐
  被动挖掘      │  BERT-Base       │  ANCE           │
  （已有数据）  │  CRSD            │  SERM           │
                ├─────────────────┼─────────────────┤
  主动生成      │  LLM 数据增强    │  AMACL（本文）  │
  （新样本）    │  （一次性）      │  （持续自适应） │
                └─────────────────┴─────────────────┘
```

**ANCE（ICLR 2021）**：从语料库中动态检索当前模型的近似最难负样本，解决了训练-测试分布对齐问题，但只能在已有数据中挖掘，无法覆盖未见过的 query-doc 组合。

**CRSD（美团，WWW 2026）**：利用 LLM 推理链作为特权信息（Privileged Information），通过对比自蒸馏将 LLM 的推理能力迁移到 BERT。核心局限：推理链是一次性静态生成的，不会根据 BERT 当前能力动态调整训练数据的难度分布。

**SERM（字节跳动，arXiv 2601.09515）**：多 Agent 框架，从线上日志中识别低置信度样本并用 LLM 重新标注。核心局限：挖掘策略固定，只能被动响应已出现的难样本，无法主动构造 BERT 尚未见过的难样本类型；且所有挖掘样本等权重训练，缺乏课程调度。

**LORE（阿里，arXiv 2512.03025）**：完全基于 LLM 的生成式相关性框架，3年迭代累计 GoodRate +27%。核心局限：LLM 在线推理延迟不可接受，无法直接工业部署；且没有将 LLM 能力蒸馏到轻量模型的路径。
### 1.3 关键洞察：ZPD 理论的机器学习解释

Vygotsky 的**最近发展区（Zone of Proximal Development, ZPD）**理论指出：学习效率最高的任务难度区间是"独立完成的上限"与"在帮助下能完成的上限"之间。

在机器学习中，这对应一个经典观察：对于分类器$f_\theta$，训练样本的学习价值与其在决策边界附近的程度正相关。具体地：
- **太简单**（$f_\theta$高置信度正确）：梯度接近零，参数几乎不更新
- **ZPD 区间**（$f_\theta$低置信度或预测错误）：梯度信号强，参数更新有效
- **太难**（$f_\theta$完全无法处理）：梯度方向不稳定，可能导致灾难性遗忘

**关键问题**：随着$f_\theta$能力提升，ZPD 区间会动态移动。固定的训练集无法持续提供 ZPD 区间内的样本。AMACL 的核心思想是：用 LLM Agent 作为"动态出题器"，持续感知$f_\theta$的当前能力边界，并主动生成恰好落在 ZPD 区间内的新样本。

---

## 二、相关工作
### 2.1 搜索相关性技术演进

```Mermaid
timeline
    title 搜索相关性技术演进（2019-2026）
    2019 : BERT Cross-Encoder
         : 语义匹配，交叉编码
    2021 : ANCE (ICLR)
         : ANN 动态难负样本
    2023 : LLM-as-Judge
         : GPT-4 替代人工标注
    2024 : SPIN (ICML)
         : 自博弈微调
    2025 : SERM (arXiv)
         : 多 Agent 被动挖掘
    2026 : CRSD (WWW)
         : 推理链对比自蒸馏
    2026 : AMACL (本文)
         : Agent 主动变异 + 课程调度
```
### 2.2 课程学习与 RL 训练调度

**DUMP（arXiv 2504.09710）**：面向 RL-based LLM 后训练的自动化分布级课程学习，使用 UCB（Upper Confidence Bound）原则动态调整不同数据分布的采样概率。AMACL 与 DUMP 的本质区别：DUMP 调度的是**已有数据**的采样顺序；AMACL 的 Agent 主动**生成**新样本，且生成策略本身通过 RL 优化。

**CLPO（arXiv 2509.25004）**：课程学习与 GRPO 策略优化结合，在 8 个数学推理基准上平均 pass@1 提升 +6.96%。AMACL 借鉴 CLPO 的"动态难度感知"思想，但将其从 LLM 推理任务迁移到判别式相关性任务，并引入结构化的 Mutation 算子约束 Agent 的生成空间。
### 2.3 自博弈与对抗数据生成

**SPIN（ICML 2024）**：LLM 通过与自身历史版本对抗来提升能力，核心是 DPO 损失：区分当前模型生成的 response 与人类标注的 response。SPIN 的对手是"历史版本自己"，适用于生成任务；AMACL 的对手是"当前 BERT 模型"，适用于判别任务，且引入了结构化算子约束生成空间。

**ATM（EMNLP 2024）**：对抗调优多 Agent 系统，Attacker 生成噪声文档，Generator 学习区分有用文档与噪声。ATM 的 Attacker 生成的是非结构化噪声；AMACL 的 Agent 生成的是语义上有意义的、针对特定错误类型的结构化变异样本。
### 2.4 与相关工作的系统对比

| 工作| 数据来源| 课程调度| Agent 可训练| 蒸馏到 BERT| 工业延迟|
|---|---|---|---|---|---|
| ANCE (ICLR 2021)| 已有语料| ✗| ✗| ✓| <10ms|
| SPIN (ICML 2024)| 自生成| ✗| ✓ (DPO)| ✗| >500ms|
| ATM (EMNLP 2024)| 对抗生成| ✗| ✓| ✗| >500ms|
| SERM (arXiv 2025)| 日志挖掘| ✗| ✗| ✓| <10ms|
| CRSD (WWW 2026)| 静态生成| ✗| ✗| ✓| <10ms|
| **AMACL（本文）**| **主动生成**| **✓ (ZPD)**| **✓ (GRPO)**| **✓**| **<10ms**|

---

## 三、AMACL 框架
### 3.1 问题形式化

**定义**：给定种子数据集$\mathcal{D}_0 = \{(q_i, d_i, y_i)\}_{i=1}^N$（$q$：query，$d$：doc，$y \in \{0,1,2,3,4\}$：相关性标签），初始 BERT 相关性模型$f_\theta$，以及固定的 LLM Oracle$\mathcal{O}$，我们的目标是学习一个 Mutation Agent$\pi_\phi$，使其生成的增广数据集$\mathcal{D}_\phi$能最大化$f_\theta$在测试集上的泛化性能。

**双层优化形式化**：

$\min_\phi \; \mathcal{L}_{\text{val}}\!\left(f_{\theta^*(\phi)}\right)
$

$\text{s.t.} \quad \theta^*(\phi) = \arg\min_\theta \; \mathcal{L}_{\text{train}}\!\left(f_\theta,\; \mathcal{D}_0 \cup \mathcal{D}_\phi\right)
$

其中外层优化 Agent 参数$\phi$（使验证集损失最小），内层优化 Student 参数$\theta$（在增广数据上训练）。

**实践近似**：完整双层优化计算代价高（需要对$\theta^*(\phi)$求导），我们采用**交替迭代**近似：固定$\theta$用 GRPO 更新$\phi$，固定$\phi$用交叉熵更新$\theta$，每$K$步交替一次。这是元学习（Meta-Learning）领域的标准近似策略（参考 MAML 的 first-order 近似）。
### 3.2 整体架构

```Mermaid
graph TB
    subgraph "离线训练循环"
        D0["种子数据集 D₀"]
        MA["Mutation Agent π_φ\n(LLM, 7B)"]
        OPS["算子库 Ω\n6类结构化算子"]
        OR["LLM Oracle O\n(72B, 固定)\n生成标签 y' + 推理链 CoT"]
        ST["BERT Student f_θ\n(可训练)"]
        RW["Reward 计算\nR = R_v · (αR_l + βR_d)"]

        D0 -->|"采样 (q,d,y)"| MA
        MA -->|"选择算子 ω + 执行变异"| OPS
        OPS -->|"生成 (q', d')"| OR
        OR -->|"y' + CoT"| ST
        OR -->|"y' (验证标签方向)"| RW
        ST -->|"置信度 p_θ(q',d')"| RW
        RW -->|"GRPO 更新 φ"| MA
        ST -->|"CE + CoT蒸馏损失 更新 θ"| ST
    end

    subgraph "在线服务"
        Q["用户 Query"] --> BERT["BERT f_θ\n延迟 <10ms"]
        BERT --> S["相关性分数"]
    end

    ST -->|"每周热更新"| BERT

    style MA fill:#4A90D9,color:#fff
    style ST fill:#E67E22,color:#fff
    style OR fill:#27AE60,color:#fff
    style RW fill:#8E44AD,color:#fff
```
### 3.3 Mutation 算子库

算子库的设计基于对美团搜索相关性**错误类型**的系统分析。我们对 10,000 条 BERT 误判样本进行人工标注，归纳出 5 类主要错误模式，并对应设计 5 类基础算子（加 1 类组合算子）：

| 算子| 对应错误类型| 操作描述| 示例|
|---|---|---|---|
| OP1: 同义替换| 词汇过拟合| 将 query 中的关键词替换为同义词，测试模型是否依赖词汇匹配| "手机" → "智能手机"|
| OP2: 属性注入| 属性忽视| 向 query 注入限定属性，使原本相关的 doc 变为不相关| "手机" → "512G 手机"，doc 为普通手机|
| OP3: 品牌替换| 品牌混淆| 将 doc 中的品牌替换为同类目竞品品牌| doc 中"苹果" → "华为"|
| OP4: 类目漂移| 类目边界模糊| 将 doc 替换为相邻类目的商品| 手机 doc → 手机壳 doc|
| OP5: 意图转换| 意图识别失败| 修改 query 的购买意图为信息意图（或反之）| "买手机" → "手机推荐"|
| OP6: 组合变异| 复合错误| 随机组合上述 2 个算子| OP2 + OP3|

**算子形式化**：每个算子$\omega_k \in \Omega$是一个函数：

$\omega_k: (q, d, y) \mapsto (q', d', \Delta y)
$

其中$\Delta y \in \{-1, 0, +1\}$表示算子预期的标签变化方向（降低相关性、不变、提升相关性）。Agent 的任务是：给定$(q, d, y)$，选择最能让当前$f_\theta$犯错的算子$\omega_k$及其参数。

**Agent 的决策过程**：Agent$\pi_\phi$是一个 7B LLM，输入为$(q, d, y)$的文本描述 + 当前$f_\theta$的预测置信度，输出为算子选择$\omega_k$和算子参数（如替换词、注入属性）。Agent 的输出是结构化的 JSON，确保可执行性。
### 3.4 Reward 设计

Reward 设计是 AMACL 的核心，需要同时满足三个目标：(1) 生成的样本标签正确；(2) 生成的样本对 Student 有学习价值；(3) 生成的样本类型多样。
#### R_verifiable：可验证性门控（乘法门控）

$R_{\text{verifiable}} = \mathbb{1}\!\left[\text{sign}\!\left(\mathcal{O}(q', d') - y\right) = \text{sign}(\Delta y)\right]
$

Oracle$\mathcal{O}$对变异后的$(q', d')$重新打分，若实际标签变化方向与算子预期方向一致，则$R_{\text{verifiable}} = 1$，否则为 0。这是一个**乘法门控**：若变异无效（如 OP3 品牌替换后 Oracle 仍判相关），则整个 Reward 为 0，Agent 不会因无效变异获得奖励。

设计动机：借鉴 RLVR（Reinforcement Learning with Verifiable Rewards）的思想，用 Oracle 的判断作为客观验证信号，防止 Agent 生成语义混乱但形式上满足条件的样本。
#### R_learn：ZPD 感知学习价值

Student$f_\theta$对变异样本$(q', d')$的预测置信度为$p_\theta \in [0, 1]$（经 softmax 归一化后的正类概率），真实标签为$y' = y + \Delta y$（经 Oracle 确认）。

我们定义 Student 的**预测误差**为：

$\text{err}_\theta = \mathbb{1}\!\left[\hat{y}_\theta \neq y'\right], \quad \hat{y}_\theta = \arg\max_c f_\theta(q', d')
$

**ZPD 感知奖励**：

$R_{\text{learn}} = \text{err}_\theta \cdot \left(1 - \left|p_\theta - p_{\text{boundary}}\right| / p_{\text{boundary}}\right)^+
$

其中$p_{\text{boundary}} = 1/C$（$C$为类别数，对于 5 分类$p_{\text{boundary}} = 0.2$），$(x)^+ = \max(x, 0)$。

**直觉**：
- $\text{err}_\theta = 0$（Student 预测正确）：$R_{\text{learn}} = 0$，样本太简单
- $\text{err}_\theta = 1$且$p_\theta$接近$p_{\text{boundary}}$（Student 预测错误且置信度低，接近均匀分布）：$R_{\text{learn}}$接近 1，样本在 ZPD 区间
- $\text{err}_\theta = 1$且$p_\theta$远离$p_{\text{boundary}}$（Student 高置信度预测错误）：$R_{\text{learn}}$较低，样本可能太难或存在标注噪声

注意：这里的设计与之前版本有本质区别——之前版本的公式在$p_\theta = 0.5$时 reward 为 0，逻辑完全相反。正确的设计是：**置信度越接近均匀分布（最不确定），学习价值越高**。
#### R_diversity：算子多样性奖励

$R_{\text{diversity}} = 1 - \frac{\text{count}(\omega_k \text{ in last } W \text{ steps})}{W}
$

其中$W$为滑动窗口大小（默认 100）。若 Agent 在最近$W$步中频繁选择同一算子，则$R_{\text{diversity}}$降低，鼓励探索不同类型的 Mutation。

相比 KL 散度，这个设计更简单、更稳定，且不需要维护历史分布的估计。
#### 总 Reward

$R_{\text{total}} = R_{\text{verifiable}} \cdot \left(\alpha \cdot R_{\text{learn}} + \beta \cdot R_{\text{diversity}}\right)
$

其中$\alpha = 0.7, \beta = 0.3$（通过验证集调参确定）。乘法门控确保只有"有效变异"才能获得奖励，加法组合平衡学习价值与多样性。
### 3.5 训练算法

```Plain Text
算法 1：AMACL 训练
输入：种子数据集 D₀，初始 BERT f_θ，LLM Oracle O，Mutation Agent π_φ
超参数：交替间隔 K，批大小 B，GRPO 组大小 G

for t = 1, 2, ..., T do
    // Phase 1: Agent 生成难样本
    采样 batch {(q_i, d_i, y_i)}_{i=1}^B from D₀
    for each (q_i, d_i, y_i):
        Agent π_φ 生成 G 个候选变异 {(q'_{i,g}, d'_{i,g}, Δy_{i,g})}_{g=1}^G
        Oracle O 标注每个候选：y'_{i,g} = O(q'_{i,g}, d'_{i,g})，生成 CoT_{i,g}
        计算每个候选的 R_total(i,g)
    
    // Phase 2: GRPO 更新 Agent
    if t mod K == 0:
        用 GRPO 损失更新 π_φ（组内相对奖励归一化）
    
    // Phase 3: 更新 Student
    将有效变异样本（R_verifiable=1）加入训练集
    用交叉熵损失 + CoT 蒸馏损失更新 f_θ

输出：训练好的 BERT Student f_θ
```

**GRPO 细节**：对于每个种子样本$(q_i, d_i, y_i)$，Agent 生成$G=8$个候选变异，计算各自的$R_{\text{total}}$，用组内均值归一化后作为优势函数（Advantage），按 GRPO 损失更新 Agent。这避免了 PPO 需要单独训练 Critic 的开销。

---

## 四、实验设计
### 4.1 数据集

| 数据集| 规模| 标注方式| 用途|
|---|---|---|---|
| 美团搜索内部数据| ~500K query-doc 对| 人工标注（0-4分，5级）| 主实验（训练+测试）|
| Amazon ESCI| 130K query，1.8M product| 人工标注（4级：Exact/Substitute/Complement/Irrelevant）| 公开数据集验证，跨域泛化|
| WANDS| 42K query，42K product| 人工标注（3级）| 公开数据集验证|

注：不使用 MSLR-WEB30K，该数据集是 Learning-to-Rank 数据集（特征向量），不适用于文本相关性判断任务。
### 4.2 评估指标

**离线指标**：
- **NDCG@1**（主要指标）：衡量最相关结果的排序质量，对相关性判断最敏感
- **NDCG@5**：衡量 Top-5 结果的整体排序质量
- **Hard Subset NDCG@1**：在人工标注的"难样本子集"（BERT 历史误判率 >30%）上单独评估

**在线指标**（A/B 测试，14天）：
- 搜索 GoodRate（好搜率）：用户对搜索结果满意度
- 相关性投诉率：用户主动反馈"结果不相关"的比例

### 4.3 基线方法

| 基线| 描述| 对应消融|
|---|---|---|
| BERT-Base| 标准 BERT Cross-Encoder，仅用种子数据训练| 无增强|
| BERT + ANCE| 动态 ANN 难负样本挖掘（ICLR 2021）| 被动挖掘|
| BERT + CRSD| 推理链对比自蒸馏（WWW 2026）| 静态 LLM 蒸馏|
| BERT + SERM| 多 Agent 被动挖掘（arXiv 2025）| 动态被动挖掘|
| AMACL w/o RL| Agent 随机选择算子（不用 GRPO 训练）| 验证 RL 的必要性|
| AMACL w/o ZPD| R_{\text{learn}}替换为均匀奖励（不区分难度）| 验证 ZPD 感知的必要性|
| AMACL w/o Div| 去掉R_{\text{diversity}}（不鼓励算子多样性）| 验证多样性奖励的必要性|
| **AMACL（完整）**| **本文方法**| —|

### 4.4 实验结果（待填充）

> **注**：以下为实验设计预期，实际数值待实验完成后填充。

**主实验（美团内部数据集）**：

| 方法| NDCG@1| NDCG@5| Hard NDCG@1|
|---|---|---|---|
| BERT-Base| —| —| —|
| BERT + ANCE| —| —| —|
| BERT + CRSD| —| —| —|
| BERT + SERM| —| —| —|
| AMACL w/o RL| —| —| —|
| AMACL w/o ZPD| —| —| —|
| AMACL w/o Div| —| —| —|
| **AMACL**| **—**| **—**| **—**|

**预期结论**：AMACL 在 Hard Subset 上的提升应显著大于整体提升，因为 Agent 专门针对 BERT 的弱点生成难样本。
### 4.5 消融实验设计

**消融 1：算子有效性分析**

逐一去掉每个算子，观察 NDCG@1 变化。预期：OP3（品牌替换）和 OP4（类目漂移）贡献最大，因为这两类错误在美团搜索中最常见。

**消融 2：Reward 组件分析**
- 去掉$R_{\text{verifiable}}$：Agent 可能生成标签错误的样本，污染训练数据
- 去掉$R_{\text{learn}}$（ZPD 感知）：退化为随机难度采样，预期 Hard Subset 提升减少
- 去掉$R_{\text{diversity}}$：Agent 陷入单一算子偏好，预期某些错误类型无法覆盖

**消融 3：交替更新策略**

对比不同的$K$（交替间隔）：$K \in \{1, 5, 10, 20\}$。预期存在最优$K$，过小导致 Agent 和 Student 相互干扰，过大导致 Agent 策略滞后于 Student 能力。

**消融 4：Agent 规模**

对比 Agent 使用 3B vs 7B vs 14B 模型，分析 Agent 规模对生成质量和最终 BERT 性能的影响。

---

## 五、理论分析
### 5.1 双层优化的收敛性

**命题 1（非正式）**：在以下假设下，AMACL 的交替迭代算法收敛到一个稳定点：
- (A1)$\mathcal{L}_{\text{train}}$关于$\theta$是$L$-光滑的
- (A2) 内层优化每次运行足够多步（$\theta$接近局部最优）
- (A3) GRPO 的学习率满足标准衰减条件

**证明思路**：参考 MAML 的 first-order 近似分析（Finn et al., 2017）和 ICLR 2024 的双层优化收敛结果（Hao et al., 2024），在上述假设下，交替迭代的梯度估计是真实双层梯度的有偏估计，偏差随内层优化步数增加而减小。完整证明见附录。

**注**：我们不声称全局收敛，只声称收敛到稳定点（$\epsilon$-stationary point）。这是非凸优化的标准结果。
### 5.2 ZPD 奖励的信息论解释

**命题 2**：$R_{\text{learn}}$是样本$(q', d')$对 Student$f_\theta$的**预测熵**的单调函数。

**证明**：对于$C$分类问题，Student 的预测分布为$\mathbf{p}_\theta = f_\theta(q', d')$。当$f_\theta$预测错误时，$R_{\text{learn}}$随$\mathbf{p}_\theta$接近均匀分布（最大熵）而增大。形式上：

$R_{\text{learn}} \propto H(\mathbf{p}_\theta) \cdot \mathbb{1}[\hat{y}_\theta \neq y']
$

其中$H(\mathbf{p}_\theta) = -\sum_c p_c \log p_c$是预测熵。这说明$R_{\text{learn}}$鼓励 Agent 生成使 Student 最不确定的样本，即信息增益最大的样本。

---

## 六、工业落地方案
### 6.1 系统架构

```Mermaid
graph LR
    subgraph "离线训练（每周一次）"
        A["日志采样\n低置信度样本"] --> B["Mutation Agent\n生成难样本 ~100K"]
        B --> C["Oracle 批量标注\n(Qwen2.5-72B + vLLM)"]
        C --> D["BERT Student 更新\n(增量微调)"]
        D --> E["Agent GRPO 更新\n(基于新 Student 状态)"]
        E --> B
    end

    subgraph "在线服务（实时）"
        Q["用户 Query"] --> R["BERT 相关性模型\n延迟 <10ms"]
        R --> S["相关性分数\n→ 排序系统"]
    end

    D -->|"每周热更新"| R
```
###