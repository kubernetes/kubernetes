# preferredimports 目录说明

`cmd/preferredimports/` 用于检查和可选修复 Go import 的推荐别名。它读取别名规则文件，解析目标 Go 源码 AST，确认符合规则的 import 使用了 Kubernetes 约定的包别名。

## 关键文件

- `preferredimports.go`：程序入口和分析器实现，支持 `-import-aliases`、`-include-path` 和 `-confirm` 参数。
- `OWNERS`：维护该工具的代码所有者信息。

## 与其他模块的关系

该工具消费 `hack/.import-aliases` 中的规则，主要服务于 e2e 相关目录和仓库编码规范。它不参与 Kubernetes 运行时逻辑，但帮助保持大规模 Go 代码中的 import 可读性和一致性。

## 开发与测试注意事项

默认模式只报告错误；加上 `-confirm` 才会重写源码。调整规则时要确认正则匹配范围足够精确，避免对无关包别名造成大面积 churn。修改 AST 重写逻辑后，应在小范围路径上验证生成 diff。
