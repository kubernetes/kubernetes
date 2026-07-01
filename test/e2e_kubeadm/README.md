# test/e2e_kubeadm 目录说明

`test/e2e_kubeadm` 包含面向 kubeadm 创建和管理集群流程的端到端测试。测试覆盖 kubeadm 配置、证书、bootstrap token、控制面节点、DNS/Proxy addon、节点加入和集群信息等场景，用于验证 kubeadm 与真实集群组件协作时的行为。

## 关键子目录/源码入口

- `e2e_kubeadm_suite_test.go`：Ginkgo 测试套件入口，注册 e2e framework 的通用 flag、集群 flag，并运行 `E2EKubeadm suite`。
- `framework.go`：封装 `Describe`，统一添加 `[sig-cluster-lifecycle] [area-kubeadm]` 标签。
- `const.go`、`util.go`、`bootstrap_signer.go`：测试常量、通用工具和 bootstrap signer 辅助逻辑。
- `*_test.go`：按 kubeadm 功能域组织的测试用例，例如 `kubeadm_config_test.go`、`kubeadm_certs_test.go`、`nodes_test.go`。

## 与 Kubernetes 其他模块的关系

这些测试复用 `test/e2e/framework` 和 `test/utils`，通过 kubeconfig 访问目标集群，并验证 `cmd/kubeadm`、kubelet、API Server、CoreDNS/kube-proxy addon、证书和 bootstrap token 相关控制器之间的集成行为。测试标签由 SIG Cluster Lifecycle 使用，常出现在发布验证和 CI 作业中。

## 开发/测试注意事项

新增用例时应使用本目录的 `Describe` 包装器保持标签一致，并避免依赖特定云厂商或易变的集群状态。测试通常需要一个由 kubeadm 管理的可用集群，运行前确认 e2e framework flag、kubeconfig、report 目录和 Ginkgo focus/skip 设置正确。涉及证书、token 或节点生命周期的改动要注意清理资源，避免影响后续测试。
