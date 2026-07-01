# gotemplate 目录说明

`cmd/gotemplate/` 是一个轻量级 `text/template` 命令行工具。它从标准输入读取模板、向标准输出写入渲染结果，并允许通过 `<key>=<value>` 参数向模板注入字符串数据。

## 关键文件

- `gotemplate.go`：程序入口和模板执行逻辑，提供 `include`、`indent`、`trim` 三个额外模板函数。
- `gotemplate_test.go`：覆盖模板渲染、参数处理和辅助函数行为。
- `OWNERS`：维护该工具的代码所有者信息。

## 与其他模块的关系

该工具通常服务于 Kubernetes 脚本和生成流程中的小型文本渲染场景，不直接依赖集群运行时代码。它依赖 Go 标准库 `text/template`，适合处理简单、可重复的模板化输出。

## 开发与测试注意事项

输入来自标准输入，参数必须是 `<key>=<value>` 形式。`include` 会读取本地文件内容，因此在脚本中使用时要明确工作目录。修改模板函数行为后，应运行 `go test ./cmd/gotemplate` 并确认既有脚本没有格式变化。
