# features 目录说明

`pkg/features` 集中定义 Kubernetes 仓库内使用的 feature gate 名称、默认状态、生命周期阶段以及与客户端 feature gate 的适配逻辑。

## 关键子目录/源码入口

- `kube_features.go`：声明 Kubernetes feature gate 常量并注册默认状态。
- `client_adapter.go`：在 Kubernetes feature gate 与 client-go feature gate 之间做适配。
- `*_test.go`：校验 feature gate 顺序、默认值和客户端适配行为。

## 与 cmd/staging/test 的关系

- `cmd` 下各组件通过组件配置或命令行参数启用/禁用这些 feature gate。
- 本目录会引用 `staging/src/k8s.io/apiserver`、`client-go`、`component-base` 及其他组件中的 feature gate 定义。
- 测试不仅覆盖本包，也会被组件启动、API 校验和集成测试间接覆盖。

## 开发/测试注意事项

- 新增 feature gate 应按文件内要求填写 owner、KEP 信息，并保持字母序。
- 修改默认状态或生命周期阶段可能影响升级和兼容性，需要确认 KEP 与发布策略。
- 建议运行本包测试，并根据 feature 影响范围运行相关组件和 API 测试。
