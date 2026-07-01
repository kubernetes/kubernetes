# util 目录说明

`pkg/util` 保存 Kubernetes 主仓库内部复用的通用工具代码，覆盖文件系统、iptables、内核信息、Pod/Node 辅助、标签和 taint/toleration 处理等能力。

关键入口和子目录：
- `iptables/` 封装 iptables 规则操作、save/restore、监控和测试 fake。
- `filesystem/`、`procfs/`、`kernel/`、`oom/` 提供节点和操作系统相关工具。
- `pod/`、`node/`、`labels/`、`taints/`、`tolerations/` 提供 Kubernetes 对象相关的轻量辅助函数。
- `async/`、`goroutinemap/`、`interrupt/`、`flock/` 提供并发、进程和锁相关工具。
- `coverage/`、`env/`、`hash/`、`parsers/`、`tail/` 等目录保存专项工具。

与 `cmd`、`staging`、`test` 的关系：
- 多个 `cmd/*` 组件会通过内部包间接使用这里的工具，但命令行入口不应放在本目录。
- 可被外部模块复用的稳定工具通常应放在 `staging/src/k8s.io/*/pkg/util`；本目录偏向主仓库内部实现细节。
- 单元测试与各工具包同目录；涉及节点、网络或系统调用的工具还可能被 e2e node 或集成测试间接覆盖。

开发和测试注意事项：
- 新增工具前先确认是否已有 staging 工具或更合适的所属目录，避免形成难以维护的杂项依赖。
- 系统调用、iptables、procfs、文件系统等代码要关注平台构建标签和 fake 覆盖。
- 工具包应保持小而无副作用，避免引入对 kubelet、apiserver 或 scheduler 的高层依赖。
