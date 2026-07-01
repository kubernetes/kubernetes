# kubectl 目录说明

`pkg/kubectl` 保存 `kubectl` 命令使用的 Kubernetes 仓库内实现代码。`cmd/kubectl` 只应作为入口，具体功能放在本目录以便测试和复用。

## 关键子目录/源码入口

- `doc.go`：说明本包与 `cmd/kubectl` 的分工。
- `cmd/convert/`：实现 `kubectl convert` 相关逻辑和已知版本导入。
- `.import-restrictions`：限制本目录可导入的包边界。

## 与 cmd/staging/test 的关系

- `cmd/kubectl` 的 `main` 入口应保持轻量，调用本目录提供的命令实现。
- kubectl 的大量通用命令代码位于 `staging/src/k8s.io/kubectl`，本目录承载 Kubernetes 仓库内仍需保留的实现或接线。
- 单元测试位于对应命令目录，端到端测试会覆盖用户可见命令行为。

## 开发/测试注意事项

- 新增命令逻辑时优先确认是否应放在 staging 的 kubectl 模块，而不是本目录。
- 注意 `.import-restrictions`，避免引入不允许的内部依赖。
- 修改用户可见输出或转换行为时，应补充单元测试，并评估 kubectl e2e 影响。
