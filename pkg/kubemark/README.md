# kubemark 目录说明

`pkg/kubemark` 提供 Kubemark 的 hollow kubelet 相关实现，用于在不运行真实节点工作负载的情况下模拟大量节点，帮助进行规模、性能和控制面压力测试。

关键入口和子目录：
- `hollow_kubelet.go` 构建模拟 kubelet 的核心行为和启动逻辑。
- `controller.go` 提供 Kubemark 运行所需的控制器侧辅助逻辑。
- 本目录体量较小，主要围绕 hollow 节点模拟，不承载真实 kubelet 的完整节点管理能力。

与 `cmd`、`staging`、`test` 的关系：
- Kubemark 的命令入口和镜像构建由仓库其他位置组装，本目录提供可复用的 hollow kubelet 代码。
- 运行时依赖 staging 中的 Kubernetes API、client 和组件基础库来模拟节点与控制面交互。
- 规模测试、性能测试和相关脚本通常位于 `test`、`cluster` 或测试基础设施目录中，本目录只保存核心实现。

开发和测试注意事项：
- 修改时要区分模拟行为和真实 kubelet 行为，避免把只适用于 Kubemark 的假设引入 `pkg/kubelet`。
- 需要关注大规模对象数量下的内存占用、watch 行为和控制面请求量。
- 测试时优先补充单元测试；涉及规模行为时应结合 Kubemark 场景或性能测试验证。
