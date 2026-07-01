# test/typecheck 目录说明

`test/typecheck` 提供跨平台 Go 类型检查工具，用于快速发现 Kubernetes 源码在多个 `GOOS/GOARCH` 组合下的类型错误。它比完整交叉编译更轻量，适合作为开发和 CI 中的早期反馈工具。

## 关键子目录/源码入口

- `main.go`：命令入口，基于 `golang.org/x/tools/go/packages` 加载包并执行类型检查；支持跨平台、并发、跳过测试代码、自定义 build tag、忽略包模式和输出耗时等参数。
- `main_test.go`：覆盖错误去重、平台参数和工具行为。
- `README`：原有说明文件，记录该工具的用途、性能收益以及 `go/types` 与 `go build` 之间可能存在的差异。
- `OWNERS`：维护者和评审归属。

## 与 Kubernetes 其他模块的关系

该工具面向整个 Kubernetes Go 模块及其依赖工作，会递归加载目标包的 imports，并覆盖 Linux、Windows、Darwin 等多个平台组合。它补充 `go test`、`go build` 和 CI 交叉编译流程，用于更快暴露平台相关类型问题。

## 开发/测试注意事项

修改检查逻辑时要保留与 `go/packages`、build tags 和测试包加载之间的兼容性。由于 `go/types` 与真实编译器并不完全等价，遇到误报时应先确认是否属于工具限制，再考虑忽略规则。运行大范围检查会消耗较多 CPU 和内存，可通过 `--platform`、`--parallel`、`--skip-test` 或 `--ignore` 缩小范围。
