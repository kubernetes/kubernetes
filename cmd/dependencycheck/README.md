# dependencycheck 目录说明

`cmd/dependencycheck/` 提供一个检查 Go 包导入依赖的小工具。它读取 `go list -json` 形式的包依赖数据，并根据命令行传入的正则规则找出被限制的 import，常用于防止仓库内某些区域依赖不该依赖的包。

## 关键文件

- `dependencycheck.go`：程序入口，解析依赖 JSON、处理 `-restrict` 与 `-exclude` 规则并报告违规 import。
- `OWNERS`：维护该工具的代码所有者信息。

## 与其他模块的关系

该工具不理解 Kubernetes 业务语义，而是消费 Go 工具链输出，服务于 Kubernetes 的依赖边界治理。它常与 vendor、staging 包和仓库内部模块边界相关的验证脚本配合使用。

## 开发与测试注意事项

运行时必须提供限制正则，例如 `-restrict`，输入文件通常由 `go list -mod=vendor -test -deps -json` 生成。修改规则时要注意它只检查输入中显式出现的依赖；如果需要覆盖传递依赖，生成输入时必须包含相应包集合。
