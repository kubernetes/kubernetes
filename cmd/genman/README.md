# genman 目录说明

`cmd/genman/` 用于为 Kubernetes 命令生成 man page。它先从组件或 kubectl 的 Cobra 命令树生成 Markdown，再通过 `go-md2man` 转换为 man 手册格式。

## 关键文件

- `gen_kube_man.go`：程序入口，支持 kube-apiserver、kube-controller-manager、kube-proxy、kube-scheduler、kubelet、kubectl 和 kubeadm。
- `OWNERS`：维护该工具的代码所有者信息。

## 与其他模块的关系

该工具导入 kubectl 与各核心组件的命令构造包，生成结果完全由命令定义、flag 和帮助文本决定。它复用 `cmd/genutils` 的输出目录检查，并与发布包、文档站点或发行物中的手册页生成流程相关。

## 开发与测试注意事项

运行时需要传入已存在的输出目录和模块名。工具会固定 `HOME`，并为 kubeadm 重置全局 flag 以减少跨命令污染。修改命令行参数或帮助文本后，应重新生成 man page 并检查格式和内容差异。
