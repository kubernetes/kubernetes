# kubemark 目录说明

`cmd/kubemark/` 是 Kubemark hollow node 二进制入口目录。Kubemark 用于创建轻量级的模拟节点，以便在较低资源成本下进行 Kubernetes 控制面规模和性能测试。

## 关键文件和子目录

- `hollow-node.go`：`main` 入口，创建并运行 hollow node 命令。
- `app/`：hollow node 命令和运行逻辑，包括与模拟 kubelet 行为相关的实现和测试。
- `OWNERS`：维护该工具的代码所有者信息。

## 与其他模块的关系

Kubemark 模拟节点侧行为，与 kube-apiserver、scheduler、controller-manager 和节点相关 API 交互，用于压测控制面而不真正运行完整容器工作负载。它与 kubelet 概念接近，但目标是测试可扩展性而不是管理真实节点。

## 开发与测试注意事项

修改 hollow node 行为时要确认不会让规模测试偏离真实节点关键信号，例如 Node、Pod 状态和心跳行为。相关变更应结合 Kubemark 测试或现有单元测试验证，并注意避免引入真实节点运行时依赖。
