# kubelet 目录说明

`pkg/kubelet` 是节点代理 kubelet 的核心实现目录，负责 Pod 生命周期管理、容器运行时交互、节点状态上报、探针、资源管理、镜像管理、卷挂载以及 kubelet 本地 HTTP 服务等节点侧逻辑。

关键入口和子目录：
- `kubelet.go`、`pod_workers.go`、`kubelet_pods.go`、`kubelet_node_status.go` 是 kubelet 主循环、Pod 同步和节点状态处理的主要入口。
- `kuberuntime/` 封装 CRI 容器运行时操作，`container/` 定义 kubelet 内部运行时接口和辅助类型。
- `cm/` 处理 CPU、内存、拓扑、设备等节点资源管理，`eviction/` 处理资源压力驱逐。
- `prober/`、`pleg/`、`status/`、`stats/` 分别负责探针执行、Pod 生命周期事件、状态缓存和节点/容器统计。
- `server/` 暴露 kubelet 本地 API，`volumemanager/` 和 `pluginmanager/` 对接卷插件与设备插件。

与 `cmd`、`staging`、`test` 的关系：
- `cmd/kubelet` 负责组装命令行、配置和依赖注入，最终启动本目录中的 kubelet 实现。
- 本目录大量依赖 `staging/src/k8s.io/*` 中的 API、client-go、component-base、CRI 和 apimachinery 组件；公共接口应优先放在 staging 模块中复用。
- 端到端和节点专项测试位于 `test/e2e_node`、`test/e2e` 等目录，单元测试主要与源码同目录放置。

开发和测试注意事项：
- kubelet 逻辑强依赖操作系统、CRI、cgroup、文件系统和网络环境，修改时要关注 Linux、Windows、Darwin/unsupported 构建标签。
- 涉及 Pod 同步、状态上报、驱逐或运行时调用的变更，通常需要补充相邻单元测试，并评估 `test/e2e_node` 覆盖。
- 避免在 kubelet 内新增对 staging 以外内部包的跨层依赖，保持节点侧边界清晰。
