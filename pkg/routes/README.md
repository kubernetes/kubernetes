# routes 目录说明

`pkg/routes` 提供 apiserver 和相关组件复用的 HTTP 路由安装逻辑，包括日志访问、OpenID 元数据和平台相关常量。

关键入口和子目录：
- `logs.go` 安装日志相关路由。
- `openidmetadata.go` 处理 OpenID Provider Configuration 与 JWKS 等服务账号发现元数据。
- `const_windows.go`、`const_other.go` 保存平台相关路径或常量差异。
- `doc.go` 提供包级说明。

与 `cmd`、`staging`、`test` 的关系：
- `cmd/kube-apiserver` 及控制面启动代码会在构造 HTTP handler 时接入这些路由。
- 本目录与 staging 中的 apiserver、serviceaccount、authentication 和 HTTP 工具类型协作。
- 路由级单元测试与源码同目录；认证发现和日志访问的整体行为可能由集成测试覆盖。

开发和测试注意事项：
- 路由路径和响应格式属于外部可见行为，修改时要考虑兼容性和客户端依赖。
- OpenID 元数据相关变更需要同时关注签名 key、issuer、service account token 和缓存行为。
- 平台常量变更应同时验证 Windows 和非 Windows 构建。
