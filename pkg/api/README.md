# api 目录说明

`pkg/api` 提供围绕核心 Kubernetes API 对象的内部辅助函数、兼容性测试工具和 legacy scheme 接入点，重点服务于 API 默认值、转换、序列化、告警和对象构造测试。

## 关键子目录/源码入口

- `legacyscheme/`：维护 Kubernetes 内部 API 的 legacy `Scheme`、`Codecs` 和参数编解码入口。
- `testing/`：包含依赖 Kubernetes API 类型的通用 API 测试、兼容性测试、fuzzer 和示例对象。
- `pod/`、`service/`、`node/`、`storage/`、`persistentvolume*` 等：提供面向具体核心资源的 util、warning 和测试构造辅助。
- `v1/`：保存面向 core/v1 外部版本对象的资源工具函数。

## 与 cmd/staging/test 的关系

- `cmd` 下组件通常不直接实现 API 细节，而是通过本目录和 `pkg/apis`、`staging/src/k8s.io/api` 中的类型与 scheme 交互。
- `staging/src/k8s.io/apimachinery`、`client-go` 等库提供 runtime、序列化和客户端基础设施，本目录承接 Kubernetes 仓库内的核心 API 约定。
- `test/integration` 和大量包内测试会依赖这里的测试 helper 与 legacy scheme。

## 开发/测试注意事项

- 修改 API 工具函数或 warning 时，要考虑默认值、转换、序列化和向后兼容测试。
- `testing/` 下内容常被跨 API 组复用，避免引入只适用于单一资源的假设。
- 涉及 `legacyscheme` 的变更风险较高，应运行相关 API、序列化和集成测试。
