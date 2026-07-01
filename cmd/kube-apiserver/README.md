# kube-apiserver 目录说明

`cmd/kube-apiserver/` 是 Kubernetes API Server 二进制入口目录。kube-apiserver 负责服务 Kubernetes 管理 API，处理认证、授权、准入、资源存储、OpenAPI/Discovery 暴露以及与 etcd 的持久化交互。

## 关键文件和子目录

- `apiserver.go`：最小化的 `main` 入口，注册日志和指标插件，创建并运行 API Server Cobra 命令。
- `app/`：命令构造、配置装配、generic apiserver、聚合层和资源安装的主要实现。
- `.import-restrictions`：该目录的 import 边界限制配置。
- `OWNERS`：维护该组件入口的代码所有者信息。

## 与其他模块的关系

该组件连接 `pkg/controlplane`、`pkg/registry`、`pkg/apis`、`staging/src/k8s.io/apiserver`、`apiextensions-apiserver` 和聚合 API Server。它输出的 API 行为会影响 `api/openapi-spec/`、`api/discovery/`、kubectl、client-go 和所有控制器。

## 开发与测试注意事项

改动启动参数、认证授权链、准入链或资源安装逻辑时，要考虑 API 兼容性和升级路径。相关测试通常覆盖单元测试、集成测试以及 OpenAPI/Discovery 验证；涉及 import 边界时还要运行导入限制验证。
