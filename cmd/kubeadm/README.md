# kubeadm 目录说明

`cmd/kubeadm/` 是 kubeadm 命令行工具入口目录。kubeadm 用于引导、加入、升级和维护 Kubernetes 集群控制面，封装证书、kubeconfig、静态 Pod、etcd、预检和 addon 等生命周期步骤。

## 关键文件和子目录

- `kubeadm.go`：`main` 入口，调用 `cmd/kubeadm/app` 并统一处理错误。
- `app/`：kubeadm 的主要实现，包含 `cmd/` 子命令、`phases/` 阶段逻辑、`apis/` 配置 API、`util/` 工具、`preflight/` 预检和 `constants/` 常量。
- `.import-restrictions`：该目录的 import 边界限制配置。
- `OWNERS`：维护该工具的代码所有者信息。

## 与其他模块的关系

kubeadm 生成并管理 kube-apiserver、kube-controller-manager、kube-scheduler、kubelet、kube-proxy 和 etcd 所需配置。它依赖 Kubernetes API 类型、client-go、组件配置、PKI 工具和 kubelet bootstrap 流程，是集群安装与升级链路的核心工具。

## 开发与测试注意事项

修改 kubeadm 流程时要特别关注升级兼容性、配置 API 版本转换、证书续期和幂等性。相关变更通常需要单元测试，并结合 kubeadm e2e 或升级测试验证。不要随意改变阶段名称、配置字段语义或已有输出格式。
