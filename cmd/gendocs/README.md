# gendocs 目录说明

`cmd/gendocs/` 是 kubectl Markdown 文档生成器。它构建 kubectl 的 Cobra 命令树，并使用 `cobra/doc` 为 kubectl 及其子命令生成 Markdown 文档。

## 关键文件

- `gen_kubectl_docs.go`：程序入口，接收输出目录参数，创建 kubectl 命令并调用 `doc.GenMarkdownTree`。
- `OWNERS`：维护该工具的代码所有者信息。

## 与其他模块的关系

生成内容来自 `k8s.io/kubectl/pkg/cmd` 中的命令定义、示例、长短描述和 flag。`cmd/genutils` 提供输出目录校验能力。kubectl 的命令行为或帮助文案变化会直接反映在生成文档中。

## 开发与测试注意事项

运行前要确保输出目录已存在；工具会固定 `HOME` 以减少环境差异。修改 kubectl 命令文档后，应通过相应的 update/verify 脚本重新生成并检查文档差异，避免把本地环境路径或临时输出写入仓库。
