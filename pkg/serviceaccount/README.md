# serviceaccount 目录说明

`pkg/serviceaccount` 实现 ServiceAccount token 的声明构造、JWT 签发与校验辅助、OpenID 发现元数据以及外部 JWT 签名插件支持。

关键入口和子目录：
- `claims.go` 构造和解析服务账号 token claims。
- `jwt.go`、`keyid.go`、`legacy.go` 处理 JWT 签名、key ID 和旧版 token 兼容逻辑。
- `openidmetadata.go` 生成服务账号 issuer 的 OpenID 发现文档和 JWKS 数据。
- `externaljwt/` 提供外部签名插件、key cache 和指标。
- `metrics.go` 记录服务账号 token 相关指标。

与 `cmd`、`staging`、`test` 的关系：
- `cmd/kube-apiserver` 在配置服务账号 issuer、签名 key 和认证器时会间接使用本目录能力。
- 本目录依赖 staging 中的认证、API、apimachinery 和组件指标库，与 `pkg/routes` 的 OpenID 路由配合工作。
- 单元测试与源码同目录；token 认证、投射和轮换行为还由集成测试及 e2e 测试覆盖。

开发和测试注意事项：
- token claims、issuer、audience、过期时间和 key 选择属于安全敏感逻辑，修改时必须补充边界测试。
- 外部签名插件路径要关注缓存失效、错误指标和密钥轮换。
- 保持签发/解析逻辑与认证器、OpenID 路由和 service account token projection 行为一致。
