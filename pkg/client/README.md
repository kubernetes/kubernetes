# client 目录说明

`pkg/client` 保存 Kubernetes 仓库内围绕客户端行为的辅助代码和测试，重点覆盖 client-go 与 Kubernetes 内部 API、watch、exec、port-forward 等交互场景。

## 关键子目录/源码入口

- `conditions/`：提供条件字段相关的客户端辅助函数。
- `tests/`：包含需要内部客户端或测试 apiserver 支撑的客户端行为测试。

## 与 cmd/staging/test 的关系

- `cmd` 下命令通常使用 `staging/src/k8s.io/client-go` 或 kubectl 工具层，本目录用于仓库内补充测试和少量 helper。
- `staging/src/k8s.io/client-go` 是主要客户端库来源，本目录不应成为新的通用客户端 API 发布面。
- `tests/` 下测试可能依赖 Kubernetes 内部类型、fake client 或测试服务器。

## 开发/测试注意事项

- 修改客户端交互测试时，应明确 fake client 与真实 apiserver 行为差异。
- 涉及 streaming、watch、remotecommand 或 port-forward 的变更，要注意协议兼容性。
- 建议运行本目录测试，并在影响 client-go 行为时同步运行 staging 相关测试。
