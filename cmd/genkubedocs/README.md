# genkubedocs 目录说明

`cmd/genkubedocs/` 是 Kubernetes 组件命令的 Markdown 文档生成器。它根据指定模块构建对应的 Cobra 命令树，为 kube-apiserver、kube-controller-manager、kube-proxy、kube-scheduler、kubelet 和 kubeadm 生成命令文档。

## 关键文件

- `gen_kube_docs.go`：程序入口，解析输出目录和模块名，并分发到各组件命令构造函数。
- `postprocessing.go`、`postprocessing_test.go`：对 kubeadm 等生成文档做后处理，便于文档站点包含。

## 与其他模块的关系

该工具直接导入各组件的 `cmd/<component>/app` 包，因此组件 flag、子命令和帮助文本变更会改变生成结果。它与 `cmd/genutils` 共用输出目录校验逻辑，也与 Kubernetes 官方文档生成流程相关。

## 开发与测试注意事项

调用时必须同时提供输出目录和模块名。新增组件文档支持时，需要在 switch 中接入对应命令构造函数，并考虑是否需要后处理。修改命令帮助文本后，应运行相应 update/verify 脚本确认生成文档稳定。
