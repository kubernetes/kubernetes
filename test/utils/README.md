# test/utils 目录说明

`test/utils` 提供 Kubernetes 测试代码共享的 Go 辅助函数，主要服务于 e2e、规模测试、密度测试和集成测试中的资源创建、等待、清理和路径处理。这里的 helper 封装常见 Kubernetes API 操作，减少测试用例重复代码。

## 关键子目录/源码入口

- `runners.go`：运行 Pod、ReplicationController、ReplicaSet、Deployment 等测试负载的核心 helper，包含资源配置结构和等待逻辑。
- `create_resources.go`、`delete_resources.go`、`update_resources.go`：批量创建、删除和更新测试资源。
- `deployment.go`、`replicaset.go`、`node.go`、`pod_store.go`：围绕常见资源类型的测试辅助逻辑。
- `conditions.go`、`density_utils.go`、`audit.go`、`admission_webhook.go`：条件判断、密度测试、审计和 admission webhook 场景的工具函数。
- `paths.go`、`tmpdir.go`、`pki_helpers.go`：源码路径、临时目录和证书/PKI 相关 helper。

## 与 Kubernetes 其他模块的关系

该包依赖 `k8s.io/api`、`k8s.io/apimachinery`、`k8s.io/client-go` 和部分 `test/utils/image` 资源，通常被 `test/e2e`、`test/e2e_node`、Kubemark/规模测试以及其他测试包导入。它处于测试层的共享基础设施位置，封装真实 clientset 与 Kubernetes API 交互。

## 开发/测试注意事项

新增 helper 时应保持接口清晰、可组合，避免把单个测试的特殊假设固化到共享工具中。涉及等待、重试和清理逻辑时要使用 context、timeout 和明确错误信息，防止测试挂起或泄漏资源。由于该目录被多类测试复用，修改后应至少运行相关单元测试，并根据影响范围补跑使用该 helper 的 e2e 或集成测试。
