# dependencyverifier 目录说明

`cmd/dependencyverifier/` 用于验证 Kubernetes Go module 依赖状态。它读取包含不期望依赖和固定版本依赖的 JSON 配置，结合 `go mod graph`、vendor 内容等信息，确认依赖图是否仍符合仓库维护策略。

## 关键文件

- `dependencyverifier.go`：核心实现，包含依赖图解析、unwanted module 检测、pinned module 检测和状态比对逻辑。
- `OWNERS`：维护该工具的代码所有者信息。

## 与其他模块的关系

该工具服务于根模块和 staging 模块的依赖管理，帮助控制 Kubernetes 对上游库的引入、移除和升级节奏。它与 `go.mod`、`vendor/`、依赖状态配置以及 CI 验证流程密切相关。

## 开发与测试注意事项

修改 `go.mod`、`go.sum` 或 vendor 后，如果依赖验证失败，应确认是期望的依赖变化还是无意引入。对 unwanted 或 pinned 依赖的配置调整需要说明原因；不要仅为了通过验证而删除状态项，应先理解依赖路径来源。
