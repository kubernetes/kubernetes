# security 目录说明

`pkg/security` 保存 Kubernetes 节点和控制面共享的安全相关辅助能力，目前主要包含 AppArmor 校验和配置处理。

关键入口和子目录：
- `doc.go` 提供包级说明。
- `apparmor/` 实现 AppArmor profile 名称、注解和字段的校验与辅助函数。
- `apparmor/validate.go`、`validate_disabled.go` 通过构建标签区分启用和不支持 AppArmor 的平台。
- `apparmor/testdata/` 保存测试所需的 profile 数据。

与 `cmd`、`staging`、`test` 的关系：
- kubelet 和 apiserver 相关启动链路会通过上层包间接使用这些安全校验能力。
- 本目录依赖 staging 中的 API 类型和字段校验工具来保持与 Kubernetes 对象语义一致。
- 单元测试位于 `apparmor/`；节点安全行为还可能由 e2e node 测试覆盖。

开发和测试注意事项：
- 安全校验变更要关注向后兼容、平台支持和 feature gate 状态。
- AppArmor 在不同内核和发行版上的可用性不同，测试应覆盖启用、禁用和不支持平台路径。
- 避免在通用安全工具中加入 kubelet 专属状态机逻辑。
