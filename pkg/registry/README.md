# registry 目录说明

`pkg/registry` 保存 Kubernetes 内置 API 资源在 apiserver 中的 REST 存储、策略、子资源和资源组装逻辑，是控制面对象持久化和准入前后策略的重要实现层。

关键入口和子目录：
- `core/`、`apps/`、`batch/`、`rbac/`、`storage/`、`networking/`、`resource/` 等子目录按 API group 组织资源策略和存储。
- 各资源下的 `strategy.go` 定义创建、更新、删除、校验前准备和字段重置等策略。
- 各 group 的 `rest/storage_*.go` 负责将资源 REST storage 装配进 apiserver。
- 资源专属 `storage/` 子目录实现子资源、特殊 REST 端点或存储行为。

与 `cmd`、`staging`、`test` 的关系：
- `cmd/kube-apiserver` 通过 apiserver 构建链路加载本目录的内置资源存储。
- 本目录大量依赖 staging 中的 API 类型、apiserver generic registry、storage、validation 和 admission 接口。
- 单元测试通常与各资源策略或 storage 同目录；集成测试和 e2e 测试会覆盖 API 行为、存储语义和兼容性。

开发和测试注意事项：
- 修改策略或存储语义会影响 API 兼容性，必须关注默认值、字段管理、状态子资源、表格输出和 declarative validation。
- 新增资源或子资源时要同步 REST storage、策略、校验、权限和相关测试。
- 避免在 registry 层引入命令行或控制器逻辑；这里应聚焦 API 存储和资源策略。
