# test/fuzz 目录说明

`test/fuzz` 是 Kubernetes 核心仓库中面向 fuzz 测试的顶层归属目录，用于集中管理与 API machinery 和测试基础设施相关的 fuzz 测试约定。当前目录主要保留 OWNERS 信息，具体 fuzz 目标通常分布在各自被测包或 staging 仓库内。

## 关键子目录/源码入口

- `OWNERS`：声明该 fuzz 测试区域的 reviewer、approver 和标签归属。
- 具体 fuzz 源码入口通常位于被测模块附近，例如 API 类型、序列化、转换、默认值、验证或解码相关包中的 `Fuzz*` 函数和 `*_test.go`。

## 与 Kubernetes 其他模块的关系

Fuzz 测试主要服务于 `k8s.io/apimachinery`、`k8s.io/apiserver`、API 类型和序列化链路，帮助发现反序列化、转换、默认值、验证和 round-trip 行为中的边界问题。该目录的归属信息与 SIG Testing、SIG API Machinery 的测试维护流程关联。

## 开发/测试注意事项

新增 fuzz 目标时优先放在被测代码邻近目录，便于共享内部 helper 和测试数据；如果需要在本目录扩展公共约定，应同步更新 OWNERS 归属。Fuzz 用例要保证可复现、资源消耗受控，避免依赖外部集群、网络或时间敏感状态。提交前可使用 Go fuzzing 命令或仓库测试脚本验证目标函数不会产生不稳定失败。
