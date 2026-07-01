# genyaml 目录说明

`cmd/genyaml/` 为 kubectl 命令树生成 YAML 格式的命令文档。它遍历 kubectl 及其一级子命令，输出命令名称、简介、描述、选项、继承选项、示例和相关命令信息。

## 关键文件

- `gen_kubectl_yaml.go`：程序入口和 YAML 生成逻辑，包含 flag 提取、长字符串处理和命令递归输出。
- `OWNERS`：维护该工具的代码所有者信息。

## 与其他模块的关系

该工具直接消费 `k8s.io/kubectl/pkg/cmd` 的 Cobra 命令定义，并复用 `cmd/genutils` 校验输出目录。生成结果可被文档站点或其他工具作为结构化 kubectl 文档输入。

## 开发与测试注意事项

运行前要确保输出目录存在。工具会固定 `HOME` 以保证 kubectl 输出稳定，并对较长字符串做多行处理以规避 YAML 库的格式问题。修改 kubectl flag、示例或说明后，应重新生成并审查 YAML 差异。
