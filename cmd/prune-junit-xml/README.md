# prune-junit-xml 目录说明

`cmd/prune-junit-xml/` 提供裁剪 JUnit XML 测试报告的工具。它会限制失败信息、跳过信息和日志输出的大小，并可把多层测试结果压缩成更适合 CI 展示的结构。

## 关键文件和子目录

- `prunexml.go`：程序入口和 XML 裁剪逻辑，支持 `-max-text-size` 和 `-prune-tests` 参数。
- `prunexml_test.go`：覆盖 XML 裁剪、失败信息合并和输出行为。
- `logparse/`：解析测试日志，帮助区分 klog 输出和真正的失败信息。
- `OWNERS`：维护该工具的代码所有者信息。

## 与其他模块的关系

该工具消费 Kubernetes 测试产生的 JUnit XML，并复用 `third_party/forked/gotestsum/junitxml` 结构。它主要服务于 CI 可读性和制品大小控制，不影响 Kubernetes 组件运行行为。

## 开发与测试注意事项

工具会原地重写传入的 XML 文件，调试时应使用副本。修改裁剪策略时要避免丢失定位失败所需的关键信息，并运行 `go test ./cmd/prune-junit-xml/...` 验证 XML 解析与输出兼容性。
