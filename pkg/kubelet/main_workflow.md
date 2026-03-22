
# Kubelet 主工作流程分析

> 基于 Kubernetes release-1.35 分支，分析路径：`pkg/kubelet/`

---

## 一、核心文件职责分布

| 文件 | 职责 |
|------|------|
| `kubelet.go` | 核心结构体、`Run()`、`syncLoop`、SyncPod 三阶段实现 |
| `pod_workers.go` | Pod 工作状态机、`podWorkerLoop`、`UpdatePod` 分发 |
| `kubelet_node_status.go` | Node 注册与节点状态同步 |
| `kubelet_pods.go` | `HandlePodCleanups` 等 Pod 清理逻辑 |
| `config/config.go` | PodConfig 多源合并、`podStorage` |

---

## 二、Kubelet struct 核心字段（kubelet.go:L1132）

`Kubelet` struct 是整个系统的"神经中枢"，集中了所有内部子系统的引用。

### Pod 生命周期驱动核心（L1153~L1196）

| 字段 | 类型 | 职责 |
|------|------|------|
| `podManager` | `kubepod.Manager` | 维护**期望状态**（desired state），是配置循环的来源 |
| `podWorkers` | `PodWorkers` | 维护**执行状态**，每个 Pod UID 对应一个独立 goroutine |

### 状态感知子系统

| 字段 | 职责 |
|------|------|
| `pleg` | 轮询容器运行时，产生 ContainerStarted/ContainerDied 等事件 |
| `eventedPleg` | 事件驱动版 PLEG（EventedPLEG feature gate） |
| `statusManager` | 将 kubelet 合成的 Pod 状态批量同步回 APIServer |
| `probeManager` | Liveness / Readiness / Startup 探针管理 |

### 资源管理子系统

| 字段 | 职责 |
|------|------|
| `evictionManager` | 节点压力驱逐 |
| `volumeManager` | 卷的 attach/mount/unmount |
| `containerManager` | cgroup、设备插件管理 |
| `allocationManager` | Pod 资源分配（含原地垂直扩缩容） |
| `imageManager` | 镜像 GC |
| `containerGC` | 容器 GC |

### 节点维护

| 字段 | 职责 |
|------|------|
| `nodeLeaseController` | 节点心跳租约（Lease） |
| `syncLoopMonitor` | 供健康检查读取，监控 syncLoop 是否卡死 |

---

## 三、Run() 方法主流程（kubelet.go:L1774）

`Run()` 接收 `updates <-chan kubetypes.PodUpdate`，是 kubelet 启动的最终入口。

```
Run(updates <-chan PodUpdate)
 │
 ├─ initializeModules()                          // L1815  启动不依赖运行时的模块
 │    ├─ metrics.Register()
 │    ├─ setupDataDirs()
 │    ├─ imageManager.Start()
 │    ├─ serverCertificateManager.Start()
 │    ├─ oomWatcher.Start()
 │    └─ resourceAnalyzer.Start()
 │
 ├─ allocationManager.Run()                      // L1827  启动资源分配管理器
 │
 ├─ go volumeManager.Run()                       // L1831  goroutine: 卷管理
 │
 ├─ go func() {                                  // L1843  goroutine: 定期同步节点状态
 │       updateRuntimeUp()
 │       wait.JitterUntil(syncNodeStatus, nodeStatusUpdateFrequency, 0.04, ...)
 │    }
 │
 ├─ go fastStatusUpdateOnce()                    // L1850  goroutine: 节点快速就绪（一次性）
 ├─ go nodeLeaseController.Run()                 // L1853  goroutine: 心跳租约续约
 ├─ go fastStaticPodsRegistration()              // L1860  goroutine: 静态 Pod Mirror 快速注册
 │
 ├─ go wait.Until(updateRuntimeUp, 5s, ...)      // L1862  goroutine: 定期检查运行时健康
 │    └─ sync.Once → initializeRuntimeDependentModules()  // 运行时首次就绪时触发（L1724）
 │         ├─ cAdvisor.Start()
 │         ├─ containerManager.Start()
 │         ├─ evictionManager.Start()
 │         ├─ pluginManager.Run()
 │         ├─ containerLogManager.Start()
 │         └─ nodeshutdownManager.Start()
 │
 ├─ statusManager.Start()                        // L1870  启动状态同步循环
 ├─ runtimeClassManager.Start()                  // L1874  RuntimeClass 同步
 ├─ pleg.Start()                                 // L1878  开始轮询 CRI（默认 1s）
 ├─ [EventedPLEG] eventedPleg.Start()            // L1882
 │
 └─ syncLoop(ctx, updates, kl)                   // L1889  主循环（阻塞，永不返回）
```

> **两阶段初始化**是关键设计：`initializeModules`（不依赖运行时）和 `initializeRuntimeDependentModules`
> （通过 `sync.Once` 在运行时首次就绪后调用）解耦了模块启动顺序，避免运行时未就绪时的启动失败。

---

## 四、syncLoop 工作机制（kubelet.go:L2500）

### 4.1 外层循环

```go
func (kl *Kubelet) syncLoop(ctx context.Context,
    updates <-chan kubetypes.PodUpdate, handler SyncHandler) {

    syncTicker         := time.NewTicker(1 * time.Second)
    housekeepingTicker := time.NewTicker(2 * time.Second)
    plegCh             := kl.pleg.Watch()

    for {
        if err := kl.runtimeState.runtimeErrors(); err != nil {
            // 运行时不健康：指数退避 100ms ~ 5s，跳过同步
            time.Sleep(duration)
            continue
        }
        kl.syncLoopMonitor.Store(kl.clock.Now())  // 更新健康检查时间戳
        if !kl.syncLoopIteration(...) { break }
        kl.syncLoopMonitor.Store(kl.clock.Now())
    }
}
```

`syncLoopMonitor` 时间戳由 `SyncLoopHealthCheck` 读取，用于监控主循环是否卡死。

### 4.2 syncLoopIteration：6 类事件的 select 多路复用（L2574）

```
syncLoopIteration 监听 6 类事件：

1. configCh    (来自 PodConfig.Updates())
   │  ADD       → handler.HandlePodAdditions(pods)
   │  UPDATE    → handler.HandlePodUpdates(pods)
   │  REMOVE    → handler.HandlePodRemoves(pods)
   │  DELETE    → handler.HandlePodUpdates(pods)    // 优雅删除作为 Update 处理
   └  RECONCILE → handler.HandlePodReconcile(pods)

2. plegCh      (来自 kl.pleg.Watch())
   │  isSyncPodWorthy(e)  → handler.HandlePodSyncs([pod])
   └  ContainerDied       → kl.cleanUpContainersInPod(...)

3. syncCh      (每秒 Ticker)
   └  kl.getPodsToSync()  → handler.HandlePodSyncs(pods)

4. livenessManager.Updates()
   └  Failure → handler.HandlePodSyncs([pod])

5. readinessManager.Updates()
   └  kl.statusManager.SetContainerReadiness(...) → handler.HandlePodSyncs([pod])

6. startupManager.Updates()
   └  kl.statusManager.SetContainerStartup(...)  → handler.HandlePodSyncs([pod])

7. containerManager.Updates()    // DRA 设备/资源变更
   └  handler.HandlePodSyncs(pods)

8. housekeepingCh  (每2秒 Ticker)
   └  handler.HandlePodCleanups(ctx)
```

> Go 的 `select` 是伪随机选择，各事件不保证优先级。所有事件最终都归结为调用 `SyncHandler` 接口方法，再分发到 `podWorkers.UpdatePod()`。

---

## 五、配置数据流（config/）

```
三个数据源 → PodConfig.mux → podStorage.Merge() → updates channel(cap=50)
                                                          │
                                                          ▼
                                              syncLoopIteration (configCh)

数据源：
  config.NewSourceFile(...)       ─── 本地静态 Pod 文件（StaticPodPath）
  config.NewSourceURL(...)        ─── HTTP 获取静态 Pod（StaticPodURL）
  config.NewSourceApiserver(...)  ─── APIServer Watch（Reflector + UndeltaStore）
       apiserver.go L37: lw = cache.NewListWatchFromClient(...)
       apiserver.go L66: r  = cache.NewReflector(lw, &v1.Pod{}, UndeltaStore, 0)
       每次 LIST/WATCH 变化 → UndeltaStore.send() → updates <- PodUpdate{Op: SET}

podStorage.Merge()(config.go:L163)：
  ├─ 根据 Op (ADD/UPDATE/DELETE/REMOVE/SET) 调用 updatePodsFunc
  ├─ checkAndUpdatePod() 判断 needUpdate / needReconcile / needGracefulDelete
  └─ 按 Incremental 模式分别发送 REMOVE/ADD/UPDATE/DELETE/RECONCILE
```

`podStorage` 是多源合并的内存真值（config.go:L125），每个 source 独立维护一个 `map[UID]*Pod`。

---

## 六、Pod 生命周期管理

### 6.1 Pod 创建（HandlePodAdditions，L2713）

```
HandlePodAdditions(pods)
 ├─ sort.Sort(PodsByCreationTime)          // 按创建时间排序，保证创建顺序
 └─ for each pod:
      ├─ podManager.AddPod(pod)            // 加入期望状态
      ├─ podCertificateManager.TrackPod()
      ├─ [if !terminating && !terminal phase]
      │    allocationManager.AddPod(...)   // 准入检查（资源分配）
      │    └─ 失败 → rejectPod()
      └─ podWorkers.UpdatePod({
             UpdateType: SyncPodCreate,
             Pod: pod, MirrorPod: mirrorPod,
         })
```

### 6.2 Pod 更新（HandlePodUpdates，L2792）

```
HandlePodUpdates(pods)
 └─ for each pod:
      ├─ podManager.UpdatePod(pod)
      ├─ [InPlacePodVerticalScaling] allocationManager.UpdatePodFromAllocation()
      └─ podWorkers.UpdatePod({UpdateType: SyncPodUpdate})
```

> DELETE 事件（优雅删除）也走 `HandlePodUpdates`（L2604），因为只是设置了 `DeletionTimestamp`，是特殊的 Update。

### 6.3 Pod 删除（HandlePodRemoves，L2940）

```
HandlePodRemoves(pods)    // REMOVE：pod 从配置源彻底消失
 └─ for each pod:
      ├─ podCertificateManager.ForgetPod()
      ├─ podManager.RemovePod(pod)         // 从期望状态移除
      ├─ allocationManager.RemovePod()
      └─ deletePod(pod)
           └─ podWorkers.UpdatePod({UpdateType: SyncPodKill})
```

---

## 七、podWorkers 状态机（pod_workers.go）

### 7.1 Pod 工作状态流转

```
        UpdatePod(SyncPodCreate / SyncPodUpdate / SyncPodSync)
                          │
                          ▼
                    ┌─────────┐
                    │ SyncPod │  ← syncPod() 被反复调用，直到 isTerminal=true 或触发终止
                    └─────────┘
                          │
          DeletionTimestamp / Eviction / Kill / terminal Phase
                          │
                          ▼
                ┌──────────────────┐
                │  TerminatingPod  │  ← syncTerminatingPod()：确保所有容器停止
                └──────────────────┘
                          │ syncTerminatingPod() 返回 nil
                          ▼
                ┌──────────────────┐
                │  TerminatedPod   │  ← syncTerminatedPod()：清理卷/目录/cgroup
                └──────────────────┘
                          │ syncTerminatedPod() 返回 nil
                          ▼
                   finished = true
                   （等 HandlePodCleanups → SyncKnownPods 清除 worker 记录）
```

### 7.2 UpdatePod 核心逻辑（L755）

```go
func (p *podWorkers) UpdatePod(options UpdatePodOptions) {
    // 1. 检查/创建 podSyncStatus
    // 2. 判断是否进入 terminating（DeletionTimestamp / 终端 Phase / Kill / 孤儿 Pod）
    // 3. 缩短 grace period（只能缩短不能延长）
    // 4. 如果 goroutine 不存在 → 创建 podUpdates channel + 启动 podWorkerLoop goroutine
    //      go p.podWorkerLoop(uid, outCh)          // L965
    // 5. status.pendingUpdate = &options           // 设置待处理更新
    // 6. podUpdates <- struct{}{}                  // 非阻塞通知（channel 容量 1）
    // 7. 如果变为 terminating 或 grace period 缩短 → status.cancelFn() 取消当前 sync
}
```

### 7.3 podWorkerLoop（L1237）

```go
func (p *podWorkers) podWorkerLoop(podUID types.UID, podUpdates <-chan struct{}) {
    for range podUpdates {                          // 等待信号
        ctx, update, canStart, canEverStart, ok := p.startPodSync(podUID)

        // 等待 PLEG 刷新运行时状态（最多 ~2s）
        status, err = p.podCache.GetNewerThan(pod.UID, lastSyncTime)

        switch update.WorkType {
        case TerminatedPod:
            err = p.podSyncer.SyncTerminatedPod(ctx, pod, status)
        case TerminatingPod:
            err = p.podSyncer.SyncTerminatingPod(ctx, pod, status, gracePeriod, podStatusFn)
        default: // SyncPod
            isTerminal, err = p.podSyncer.SyncPod(ctx, updateType, pod, mirrorPod, status)
        }

        // 状态转换
        p.completeWork(podUID, phaseTransition, err)
    }
}
```

> **每个 Pod UID 对应独立 goroutine**，`podUpdates` channel 容量为 1。新 update 总是覆盖旧的 `pendingUpdate`，保证最终一致而非处理每个中间状态（L986）。

---

## 八、SyncPod 三阶段实现

### 8.1 SyncPod（kubelet.go:L1941）— 运行阶段

```
SyncPod(ctx, updateType, pod, mirrorPod, podStatus)
 ├─ generateAPIPodStatus(pod, podStatus)           // 生成 API 格式状态
 ├─ [if phase == Succeeded/Failed] → isTerminal=true, return
 ├─ statusManager.SetPodStatus(...)                // 上报当前状态
 ├─ [if network not ready && !hostNetwork] → return error
 ├─ secretManager.RegisterPod / configMapManager.RegisterPod
 ├─ containerManager.NewPodContainerManager().EnsureExists(pod)  // 创建 cgroup
 ├─ tryReconcileMirrorPods(...)                    // 静态 Pod mirror 对账
 ├─ makePodDataDirs(pod)                           // 创建 Pod 数据目录
 ├─ volumeManager.WaitForAttachAndMount(ctx, pod)  // 等待卷 attach/mount 就绪
 ├─ getPullSecretsForPod(pod)                      // 获取镜像拉取 Secret
 ├─ probeManager.AddPod(ctx, pod)                  // 注册探针
 └─ containerRuntime.SyncPod(...)                  // 调用 CRI：创建/重启容器
```

### 8.2 SyncTerminatingPod（kubelet.go:L2182）— 终止阶段

```
SyncTerminatingPod(ctx, pod, podStatus, gracePeriod, podStatusFn)
 ├─ generateAPIPodStatus / statusManager.SetPodStatus
 ├─ probeManager.StopLivenessAndStartup(pod)       // 停止存活/启动探针
 ├─ killPod(ctx, pod, runningPod, gracePeriod)      // 停止所有容器
 ├─ probeManager.RemovePod(pod)                    // 移除就绪探针
 ├─ containerRuntime.GetPodStatus(...)             // 验证容器确已停止
 ├─ [DynamicResourceAllocation] UnprepareDynamicResources
 └─ statusManager.SetPodStatus(...)                // 最终状态上报
```

### 8.3 SyncTerminatedPod（kubelet.go:L2339）— 清理阶段

```
SyncTerminatedPod(ctx, pod, podStatus)
 ├─ generateAPIPodStatus / statusManager.SetPodStatus
 ├─ volumeManager.WaitForUnmount(ctx, pod)          // 等待卷卸载
 ├─ wait.PollUntilContextCancel → podVolumesExist() == false  // 轮询卷清理完成
 ├─ secretManager.UnregisterPod / configMapManager.UnregisterPod
 ├─ containerManager.UpdateQOSCgroups / pcm.Destroy(pod)       // 删除 cgroup
 └─ statusManager.TerminatePod(pod)                // 标记 Pod 完全终止
```

---

## 九、主要 Goroutine 汇总

| Goroutine | 启动位置 | 周期/触发 | 职责 |
|-----------|----------|-----------|------|
| `volumeManager.Run` | `Run() L1831` | 持续运行 | 卷 attach/detach/mount/unmount |
| `syncNodeStatus` | `Run() L1843` | `nodeStatusUpdateFrequency` (+4% jitter) | 节点状态同步到 APIServer |
| `fastStatusUpdateOnce` | `Run() L1850` | 100ms 轮询，一次性 | 节点快速就绪检测 |
| `nodeLeaseController.Run` | `Run() L1853` | 0.25x lease duration | 心跳租约续约 |
| `fastStaticPodsRegistration` | `Run() L1860` | 一次性 | 静态 Pod Mirror 快速注册 |
| `updateRuntimeUp` | `Run() L1862` | 5s | 检查运行时健康，首次就绪触发二阶段初始化 |
| `statusManager`（内部） | `Run() L1870` | 内部定时 | 批量上报 Pod 状态 |
| `pleg.Start` | `Run() L1878` | 1s relist | 轮询 CRI，生成容器生命周期事件 |
| `syncLoop` | `Run() L1889` | select 多路复用 | **主循环**：分发所有 Pod 操作 |
| `podWorkerLoop` per Pod | `UpdatePod() L965` | channel 信号触发 | 每个 Pod 独立状态机 goroutine |
| `evictionManager` | `initializeRuntimeDependentModules L1750` | 10s | 节点资源压力监控与驱逐 |
| `pluginManager.Run` | `initializeRuntimeDependentModules L1764` | 持续运行 | CSI/设备插件注册 |
| `containerLogManager` | `initializeRuntimeDependentModules L1754` | 持续运行 | 日志轮转 |

---

## 十、完整调用链总结

```
                      ┌─────────────────────────────────────┐
                      │         数据来源（配置层）            │
                      │                                     │
                      │  APIServer Watch (Reflector)         │
                      │  StaticPodFile / StaticPodURL        │
                      └──────────────┬──────────────────────┘
                                     │ PodUpdate
                                     ▼
                          podStorage.Merge()
                          updates channel (cap=50)
                                     │
                                     ▼
                      ┌─────────────────────────────────────┐
                      │          syncLoopIteration           │
                      │  (select: config/pleg/sync/probe/   │
                      │           housekeeping)              │
                      └──────────────┬──────────────────────┘
                                     │
                         HandlePodAdditions / Updates / Removes
                                     │
                                     ▼
                          podWorkers.UpdatePod()
                          创建 or 唤醒 per-Pod goroutine
                                     │
                                     ▼
                      ┌─────────────────────────────────────┐
                      │         podWorkerLoop goroutine      │
                      │                                     │
                      │  podCache.GetNewerThan()             │  ← 等待 PLEG 刷新（最多 2s）
                      │           │                         │
                      │           ▼                         │
                      │  SyncPod / SyncTerminatingPod /      │
                      │  SyncTerminatedPod                  │
                      └──────────────┬──────────────────────┘
                                     │
                          containerRuntime.SyncPod()          ← CRI gRPC 调用
                          statusManager.SetPodStatus()        ← PATCH /pods/.../status

─────────────────────────────────────────────────────────────

PLEG（1s 轮询 CRI）
  → plegCh → HandlePodSyncs → UpdatePod(SyncPodSync) → 同上流程

Liveness 探针失败
  → livenessManager.Updates() → HandlePodSyncs → 重触发 SyncPod

Housekeeping（每 2s）
  → HandlePodCleanups → SyncKnownPods 清理 finished workers
  → probeManager.CleanupPods / cleanupOrphanedPodDirs
```

---

## 十一、架构设计要点

### 1. 双状态真值分离
`podManager`（期望状态）与 `podWorkers`（执行状态）严格解耦。前者来自配置源，后者是实际运行驱动。两者分离避免了竞态条件，是 kubelet 架构的核心设计（参见 `kubelet.go:L1153` 注释）。

### 2. Per-Pod goroutine + channel(1)
每个 Pod UID 对应一个独立的 `podWorkerLoop` goroutine，`podUpdates` channel 容量为 1。多次快速 update 只触发一次处理（取最新状态），不堆积中间状态，保证最终一致性（`pod_workers.go:L986`）。

### 3. PLEG 等待机制
每次 sync 前调用 `podCache.GetNewerThan(uid, lastSyncTime)` 等待 PLEG 刷新，最多等待约 2s，确保每次 SyncPod 基于最新的容器运行时状态（`pod_workers.go:L1279`）。

### 4. 两阶段初始化
- **第一阶段** `initializeModules`（`kubelet.go:L1671`）：启动不依赖运行时的模块（imageManager、oomWatcher 等）
- **第二阶段** `initializeRuntimeDependentModules`（`kubelet.go:L1724`）：通过 `sync.Once` 在运行时首次就绪后触发，启动 evictionManager、pluginManager 等

### 5. Grace Period 单调递减
Grace period 只能缩短、不能延长（`pod_workers.go:L999` 的 `calculateEffectiveGracePeriod`）。Eviction 可以强制覆盖为更短的 grace period，但任何路径都不能延长已设置的 grace period。

### 6. 运行时不健康的指数退避
syncLoop 外层在运行时不健康时做指数退避（base=100ms, max=5s, factor=2），避免无效重试消耗资源（`kubelet.go:L2513`）。