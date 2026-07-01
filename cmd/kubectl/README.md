# kubectl 目录说明

`cmd/kubectl/` 是 kubectl 客户端二进制入口目录。kubectl 是 Kubernetes 的主要命令行客户端，用于查看、创建、修改和调试集群资源。

## 关键文件

- `kubectl.go`：`main` 入口，初始化客户端认证插件、设置日志 verbosity，创建默认 kubectl 命令并运行。
- `OWNERS`：维护该入口目录的代码所有者信息。

## 与其他模块的关系

实际命令实现主要位于 `k8s.io/kubectl/pkg/cmd`，并大量依赖 cli-runtime、client-go、apimachinery、RESTMapper 和 Kubernetes API 类型。kubectl 与 kube-apiserver 的 API 行为、OpenAPI/Discovery 信息和认证插件体系密切相关。

## 开发与测试注意事项

入口目录应保持轻量，命令行为变更通常应在 kubectl 包内完成。修改参数解析、日志或插件初始化时，要确认不会破坏命令构造期间的 klog 使用、kuberc 解析和插件发现流程。CLI 文案变化还会影响 `cmd/clicheck` 和文档生成器。
