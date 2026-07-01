# credentialprovider 目录说明

`pkg/credentialprovider` 提供镜像仓库凭据解析、匹配、缓存和插件接入能力，供 kubelet 等组件为拉取容器镜像获取认证信息。

## 关键子目录/源码入口

- `provider.go`、`keyring.go`、`config.go`：定义凭据 provider、Docker keyring、配置结构和匹配逻辑。
- `plugin/`：实现外部 credential provider 插件配置、调用、缓存和指标。
- `secrets/`：从 Kubernetes Secret 构造镜像拉取凭据。

## 与 cmd/staging/test 的关系

- kubelet 入口在 `cmd/kubelet`，实际镜像拉取凭据逻辑会通过 kubelet 包使用本目录能力。
- 与 CRI、client-go Secret 类型和组件指标库交互，但本目录本身不是 staging 发布 API。
- 单元测试覆盖 keyring、provider、plugin 配置和 Secret 解析；端到端测试可能覆盖真实镜像拉取路径。

## 开发/测试注意事项

- 修改凭据匹配时，要注意 registry host、path 前缀、通配符和默认凭据顺序。
- 插件相关变更要关注 exec 超时、缓存 TTL、错误处理和指标稳定性。
- 避免在日志或错误中泄露认证信息；测试中也应使用假凭据。
