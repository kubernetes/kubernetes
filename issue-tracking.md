# 个人 Issue 与 PR 状态追踪

> 更新时间：2026-09-23。仓库：kubernetes/kubernetes（KEP 条目位于 kubernetes/enhancements）。
> 本文件为个人工作笔记，未纳入 git 提交。

## 一、`#137310` 与 rescore 的关系（2026-09-23 核实）

**结论：正交互补，不构成替代，`#137310` 仍有必要。**

两者作用于调度开销的不同维度：

| 维度 | rescore（KEP-5598） | `#137310` 缓存 |
|---|---|---|
| 优化对象 | 减少需要重打分的**节点数** | 降低每次计算的**单位成本** |
| 生效条件 | 同签名 pod 连续入批 | 对所有 pod 生效 |
| 覆盖阶段 | Score（及 PreScore） | Filter 与 Score |
| 首个 pod | 无批状态可复用，无收益 | 有收益 |
| 非 group / 签名变化的 pod | 无收益 | 有收益 |
| 已放置后需重算的受影响节点 | 减少重算范围 | 加速每次重算 |

关键事实：

1. **Rescore 扩展点尚未实现。** 当前 master 仅有 `rescoreHintedNode`
   （`pkg/scheduler/framework/runtime/batch.go:258`），只对单个
   `lastChosenNodeInfo` 调 `RunRawScorePlugins`，属节点局部特例。拟议的
   `RescorePlugin` 接口在 `kubernetes/enhancements#6411` 中仅为文档提案
   （2026-09-22 开启，OPEN），全仓 `rg RescorePlugin` 无命中。
2. **KEP-5598 自认默认配置下零收益。** 由于默认调度器 profile 启用了
   preferred `PodTopologySpread` 约束（组感知），当前 opportunistic batching
   在默认配置下不产生任何收益。Rescore 落地前，`#137310` 所优化的两个
   插件恰好得不到 rescore 的任何帮助。
3. **`#137310` 的缓存对象与 rescore 不重叠。** 该 PR（2,700 行，已因
   `lifecycle/rotten` 被机器人关闭，`mergedAt: null`）缓存
   `ExistingPodAffinityTermDetailedState` /
   `IncomingPodAffinityTermDetailedState`，即按
   (namespace, name, node) 记忆化亲和项匹配——namespace 标签查询、pod
   标签匹配、topology key 求值；`filtering_map.go` 另行优化 Filter 阶段；
   PodTopologySpread 侧缓存 score map。`RemovePod` 支持增量维护。
4. **即便 Rescore 落地，缓存反而更有价值。** 定向 rescore 后，剩余开销
   集中于受影响节点的重算，而这正是缓存所加速的部分。

推进建议（待定）：

- 重开前先 rebase（原分支基线已大幅落后），并考虑与 `enhancements#6411`
  对齐——Rescore 扩展点定稿后，插件的增量计算接口可能与缓存插件的
  汇报粒度产生交互。
- `#137310` 关闭原因为机器人清理，非评审否决，重开无程序障碍。

## 二、`#137306` 与 `#137654` 的关系（2026-09-23 核实）

- `#137306`（2026-02-28）：零第三方参与，跟踪 KEP `enhancements#5953`
  已由本人于 2026-03-12 以 `completed` 关闭，`lifecycle/rotten` 已 76 天。
  判定为已放弃的重复提报。处置：以 duplicate 关闭，指向 `#137654`。
- `#137654`（2026-03-12）：唯一活跃线索。所有声称的实现均已终止：
  `kubernetes/autoscaler#9461` CLOSED 未合并（04-22）、
  `kubernetes/autoscaler#9523` CLOSED 未合并（05-08）、
  `bwsalmon/kubernetes#3` 停留在个人 fork（最后更新 03-17）、
  `#137310` 被机器人关闭（07-06）。
  注意：x13n 的两个原型位于 Cluster Autoscaler（绕过调度器插件），
  并非 kube-scheduler 代码，不能视为调度器侧问题的解决。
  `x13n` 于 06-19 执行 `/cc @tetianakh` 后无回应，静止 96 天。
  处置：不应关闭；考虑重开 `#137310` 并在 issue 中说明调度器侧仍空缺。

## 三、开放条目快照（2026-09-23）

### PR（3 个）

| PR | 更新 | 状态 | 阻塞 |
|---|---|---|---|
| `kubernetes#142315`（PodGroup 节点维度诊断） | 09-22 | OPEN / MERGEABLE / size/XXL | `needs-ok-to-test`，CI 未启动 |
| `kubernetes#136217`（state 克隆优化） | 09-21 | OPEN / MERGEABLE / size/M | pacoxu 08-28 评审「lazy clone LGTM, only nits in test」；nit 已处理，`/release-note-none` 已发；仍缺 `lgtm`/`approved`、`priority`、`triage` |
| `kubernetes#141137`（cache dumper JSON） | 08-31 | OPEN / MERGEABLE / size/L | `reviews` 为空数组，`/cc @pohly @sanposhiho` 后 23 天无响应 |

### Issue（7 个）

| Issue | 更新 | 静止 | 判断 |
|---|---|---|---|
| `kubernetes#141025`（PodGroup 诊断不足） | 09-23 | 0 天 | 活跃；`#142315`、`#141860`、`#140670` 围绕它展开 |
| `enhancements#5951`（KEP：批处理） | 09-03 | 20 天 | `lifecycle/stale`；驱动 issue `#136221` 已于 09-23 关闭为 not_planned，KEP PR `#5952` 已于 08-31 未合并关闭，已孤立 |
| `kubernetes#141510`（DRA 按租户限制） | 08-25 | 29 天 | **待回复 pohly**；其立场：纯源码改动无需 KEP，归 SIG Scheduling，不反对，但需配套自定义 autoscaler；另提 `structured` 代码归属需清理 |
| `kubernetes#139369`（签名级失败缓存） | 08-04 | 50 天 | 受让人 `jianzhangbjz`；等待 `#140355`（antekjb，按签名分组同质 pod 子组）合入；该 PR 09-22 仍在更新，Argh4k 的 09-18 `lgtm` 因新提交被 prow 撤销，另带 `do-not-merge/hold`；macsko 已承诺合入后协同 |
| `kubernetes#139640`（cache debugger JSON） | 07-31 | 54 天 | 无受让人；与 `#141137` 为同一议题，PR 零评审导致 issue 停滞 |
| `kubernetes#137306`（InterPodAffinity 缓存） | 07-09 | 76 天 | `lifecycle/rotten`；与 `#137654` 重复，建议 duplicate 关闭 |
| `kubernetes#137654`（缓存机制） | 06-19 | 96 天 | 唯一活跃线索，见第二节 |

### 本日已关闭

- `kubernetes#136221`（异步 moveAllToActiveOrBackoffQueue）→ `not_planned`。
  关联：PR `#137265`（未合并关闭）、KEP PR `enhancements#5952`（未合并关闭）。
- `kubernetes#137707`（OpportunisticBatching 冗余同步 Filter）→ `completed`。
  缺陷经 `e74bd42838d`（经 `#140289` 合并）的 rescore 重设计消除；
  未决议题（非 group pod 关闭批处理、Rescore 扩展点）已在关闭说明中记录，
  Rescore 扩展点由 `enhancements#6411` 跟踪。

### `#139369` 的依赖链（2026-09-23 补充）

- `macsko` 于 08-03 明确：「that's our plan for PodGroups, but first we need
  https://github.com/kubernetes/kubernetes/pull/140355 to land」。
- 08-04 约定协同，`macsko` 原话「Reach out to me once it happens, as I may forget」。
- `#140355`（antekjb，`Add grouping Pods into sorted homogeneous sub-groups`，
  size/XXL）按签名将 pod 分入同质子组，使 PodGroup 能高效利用 opportunistic
  batching——即 `#139369` 签名缓存的基础设施。
- 其 09-18 曾获 Argh4k `lgtm`，09-22 因新提交被 prow 撤销；当前标签含
  `do-not-merge/hold`、`needs-priority`、`needs-triage`。
- **动作**：监控 `#140355` 合入，合入后立即在 `#139369` 中 @macsko 触发协同。

## 四、按优先级的待办

1. **回复 pohly（`#141510`）** — 唯一有维护者正面回应且等待答复的条目，静止 29 天。
2. **为 `#141137` 争取评审** — 三周零评审；`/cc` 失效，建议经 SIG Slack 或改请 `macsko`。
3. **为 `#142315` 争取 `/ok-to-test`** — CI 完全未启动；`helayoty` 已加 WG 标签，是合理求助对象。
4. **以 duplicate 关闭 `#137306`** — 依据充分（零参与、KEP 已关、rotten 76 天）。
5. **决定 `#137654` / `#137310` 的重启** — 先 rebase，再对齐 `enhancements#6411`。
6. **处置 `enhancements#5951`** — 与 `#136221`、`#5952` 保持一致（一并关闭）或重启 KEP。

## 五、结构性观察

- 7 个 issue 中 6 个仍带 `needs-triage`；3 个 PR 全部带 `needs-priority` 与
  `needs-triage`。绝大多数条目尚未进入 SIG 正式处理队列。
- 唯一进入活跃轨道的是 `#141025` 一线，同时牵动 `#142315`、`#141860`、
  `#140670`，已由 `helayoty` 补加 `wg/workload-aware-scheduling`。
- 三个 PR 中没有一个拿到 `lgtm`，其中两个连 CI 都未启动或未出结论。
