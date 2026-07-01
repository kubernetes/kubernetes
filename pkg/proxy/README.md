# proxy 目录说明

`pkg/proxy` 是 kube-proxy 的核心实现目录，负责监听 Service、EndpointSlice 和 Node 变化，并在不同平台或后端上维护服务转发规则。

关键入口和子目录：
- `types.go`、`servicechangetracker.go`、`endpointschangetracker.go`、`endpointslicecache.go` 定义代理共享模型和变更跟踪。
- `iptables/`、`ipvs/`、`nftables/`、`winkernel/` 分别实现不同数据平面后端。
- `config/` 监听 apiserver 对象并驱动 proxier 同步。
- `healthcheck/`、`metrics/`、`conntrack/`、`util/` 提供健康检查、指标、连接跟踪和网络辅助能力。
- `kubemark/` 提供 hollow proxy 相关实现。

与 `cmd`、`staging`、`test` 的关系：
- `cmd/kube-proxy` 负责解析配置、构造依赖并启动本目录中的 proxier。
- 配置 API 位于本目录 `apis/config`，同时依赖 staging 中的 API machinery、client-go 和 component-base。
- 后端行为由同目录单元测试覆盖，跨节点服务连通性由 `test/e2e` 网络相关用例覆盖。

开发和测试注意事项：
- 修改规则生成或同步逻辑时，需要分别考虑 iptables、IPVS、nftables 和 Windows kernel 后端的行为差异。
- 网络规则变更要关注双栈、NodePort、ExternalTrafficPolicy、拓扑感知和连接跟踪清理。
- 涉及配置字段时要同步默认值、转换、校验和 roundtrip 测试。
