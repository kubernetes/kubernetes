# clicheck 目录说明

`cmd/clicheck/` 构建一个用于检查 kubectl 命令行约定的小工具。它实例化完整的 kubectl 命令树，并运行 `k8s.io/kubectl/pkg/cmd/util/sanity` 中的命令和全局规则，检查帮助文本、参数组织、命令命名等 CLI 约定。

## 关键文件

- `check_cli_conventions.go`：程序入口，创建 kubectl 命令并输出所有检查失败项。
- `OWNERS`：维护该工具的代码所有者信息。

## 与其他模块的关系

该工具直接依赖 `k8s.io/kubectl/pkg/cmd` 中的 kubectl 命令定义，因此 kubectl 子命令、flag 或帮助文本变化都可能影响检查结果。它通常服务于 `hack/verify-*` 类脚本和 CI，帮助保持 Kubernetes CLI 的一致性。

## 开发与测试注意事项

调整 kubectl 命令结构或帮助文案后，如果该工具失败，应优先修正命令定义而不是放宽规则。可用 `go run ./cmd/clicheck` 在仓库根目录快速复现；输出中的每条错误都对应一个需要人工确认的 CLI 约定问题。
