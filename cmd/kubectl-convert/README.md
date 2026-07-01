# kubectl-convert 目录说明

`cmd/kubectl-convert/` 是 `kubectl convert` 独立二进制入口目录。该工具用于把 Kubernetes 对象清单在不同 API 版本之间转换，便于用户迁移 YAML/JSON 配置。

## 关键文件

- `kubectl-convert.go`：程序入口，创建 kubeconfig flag、Factory 和 convert 命令，并通过 component-base CLI 运行。
- `OWNERS`：维护该工具入口的代码所有者信息。

## 与其他模块的关系

转换逻辑位于 `pkg/kubectl/cmd/convert`，并依赖 cli-runtime、kubectl util、RESTMapper、scheme 和 Kubernetes API 类型注册。它与 kube-apiserver 的版本转换语义相关，但作为客户端工具运行，不直接启动控制面组件。

## 开发与测试注意事项

修改转换行为时要验证输入输出对象版本、字段保留和错误提示。需要关注已弃用 API 的迁移体验，以及与 kubectl 共享的 kubeconfig、认证插件和命令 flag 行为是否一致。
