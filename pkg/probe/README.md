# probe 目录说明

`pkg/probe` 提供 kubelet 使用的通用探针实现，包括 exec、HTTP、TCP 和 gRPC 探测，用于判断容器存活、就绪和启动状态。

关键入口和子目录：
- `probe.go` 定义探测结果、接口和基础类型。
- `exec/`、`http/`、`tcp/`、`grpc/` 分别实现不同协议或执行方式的探测器。
- `util.go` 和平台相关 `dialer_*` 文件提供超时、网络拨号和共享辅助逻辑。
- `doc.go` 说明包级用途。

与 `cmd`、`staging`、`test` 的关系：
- `cmd/kubelet` 启动 kubelet 后，探针最终由 `pkg/kubelet/prober` 调用本目录实现。
- 本目录使用 staging 中的 API 类型和组件工具库，但自身保持为节点侧通用探测能力。
- 单元测试与各协议实现同目录；探针的端到端行为由 kubelet 和 e2e 测试间接覆盖。

开发和测试注意事项：
- 探针超时、重试、错误分类和上下文取消会直接影响 Pod 状态，修改时必须补充协议级测试。
- 网络相关代码要兼顾 Windows 和非 Windows 平台实现差异。
- 不要在通用探针层引入 kubelet 状态机逻辑，Pod 语义应留在调用方处理。
