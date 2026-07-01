# genutils 目录说明

`cmd/genutils/` 提供文档生成命令共享的辅助函数。目前主要负责把传入的输出目录转换为绝对路径，并确认该路径已经存在且是目录。

## 关键文件

- `genutils.go`：包含 `OutDir`，供多个生成器统一校验输出目录。
- `genutils_test.go`：覆盖输出目录解析和错误场景。

## 与其他模块的关系

`cmd/gendocs`、`cmd/genkubedocs`、`cmd/genman`、`cmd/genyaml` 等生成器都依赖这里的输出目录处理逻辑。它不包含 Kubernetes 业务逻辑，但影响文档生成工具对路径错误的处理方式。

## 开发与测试注意事项

修改 `OutDir` 时要注意保持调用方期望：返回绝对路径并以 `/` 结尾，且不自动创建目录。变更后应运行 `go test ./cmd/genutils`，并留意依赖该包的生成器是否需要同步调整。
