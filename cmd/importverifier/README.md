# importverifier 目录说明

`cmd/importverifier/` 用于验证 Go 包是否遵守仓库中的 import 限制规则。它读取 `go list` 输出和 `.import-restrictions` 配置，检查指定目录树内的包是否导入了被禁止的依赖。

## 关键文件

- `importverifier.go`：程序入口和核心规则实现，包含限制配置解析、路径匹配和违规 import 收集逻辑。
- `OWNERS`：维护该工具的代码所有者信息。

## 与其他模块的关系

Kubernetes 使用该工具维护源码层次边界，例如防止低层库依赖高层组件，或限制 staging、cmd、pkg 之间的不合理耦合。`cmd/kube-apiserver`、`cmd/kubeadm` 等目录下的 `.import-restrictions` 会被它消费。

## 开发与测试注意事项

新增依赖前应先确认所在目录的 import 限制。验证失败时，优先调整代码结构或移动公共逻辑，而不是扩大允许列表。只有在架构边界确实变化且经过评审时，才应修改 `.import-restrictions`。
