# securitycontext 目录说明

`pkg/securitycontext` 提供 Pod 和 Container SecurityContext 的访问、合并和平台相关辅助逻辑，供 kubelet、准入和运行时相关代码复用。

关键入口和子目录：
- `accessors.go` 定义从 Pod、Container 和 SecurityContext 中读取安全字段的访问器。
- `util.go` 及 `util_linux.go`、`util_windows.go`、`util_darwin.go` 提供平台相关的安全上下文处理。
- `fake.go` 为测试提供辅助实现。
- `doc.go` 提供包级说明。

与 `cmd`、`staging`、`test` 的关系：
- kubelet 由 `cmd/kubelet` 启动后，会在容器运行时配置生成过程中使用这些安全上下文工具。
- 本目录围绕 staging 中的 core API 类型工作，不定义新的外部 API。
- 单元测试与源码同目录；安全上下文的节点运行效果还需依赖 kubelet 和 e2e node 测试覆盖。

开发和测试注意事项：
- 修改字段合并或默认逻辑会影响容器运行时参数，应覆盖 Pod 级和 Container 级组合场景。
- 平台差异很重要，Linux、Windows 和其他平台的构建标签路径都需要保持可编译。
- 不要在工具层直接调用运行时或 apiserver，调用方负责上下文和副作用。
