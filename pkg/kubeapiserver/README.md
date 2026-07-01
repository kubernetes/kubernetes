# kubeapiserver 目录说明

`pkg/kubeapiserver` 放置 kube-apiserver 专用但不属于通用 apiserver 框架的代码，包括认证授权选项、准入插件配置和默认存储工厂设置。

## 关键子目录/源码入口

- `default_storage_factory_builder.go`：构造 kube-apiserver 默认 storage factory 配置、资源编码和 etcd 覆盖规则。
- `options/`：认证、授权、准入、服务证书和插件相关命令行/配置选项。
- `authenticator/`、`authorizer/`：kube-apiserver 认证和授权配置辅助。
- `admission/`：kube-apiserver 准入插件初始化和配置。

## 与 cmd/staging/test 的关系

- `cmd/kube-apiserver` 是启动入口，主要调用本目录和 `pkg/controlplane` 完成 kube-apiserver 配置。
- 通用 server、认证授权接口和 admission 框架来自 `staging/src/k8s.io/apiserver`，本目录只承载 Kubernetes 发行组件特有逻辑。
- 包内测试覆盖选项和配置，集成测试覆盖完整 apiserver 启动、认证授权和存储路径。

## 开发/测试注意事项

- 修改默认存储版本或资源编码时，要考虑升级、降级和 etcd 数据兼容。
- 认证授权选项变更会影响安全边界，应覆盖允许、拒绝、重载和错误路径。
- 建议运行本目录测试，并根据改动运行 kube-apiserver 集成测试。
