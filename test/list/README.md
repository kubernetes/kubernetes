# test/list 目录说明

`test/list` 提供一个小型 Go 工具，用于静态解析 Kubernetes 测试源码并列出将运行的单元测试和 Ginkgo 测试名称。它帮助测试基础设施、脚本和开发者了解测试集合，而不需要真正执行测试。

## 关键子目录/源码入口

- `main.go`：命令入口，解析 Go AST，收集 `Test*` 单元测试以及 Ginkgo `Describe`、`Context`、`It` 等结构生成的测试路径；支持 `--json`、`--dump`、`--warn` 等参数。
- `main_test.go`：覆盖测试名称收集和 AST 解析行为。

## 与 Kubernetes 其他模块的关系

该工具面向整个 `k8s.io/kubernetes` 源码树工作，理解 Kubernetes e2e 和单元测试中常见的 Ginkgo 写法。它不直接依赖运行中集群，而是为 `hack/`、CI 或测试分析流程提供测试清单能力。

## 开发/测试注意事项

修改解析规则时要注意 Ginkgo API 和 Kubernetes 测试包装函数的变化，避免遗漏或错误展开动态测试名称。新增行为应补充 `main_test.go` 中的样例，优先保持输出稳定，特别是 JSON 字段和测试路径格式。该工具做静态分析，无法完全还原运行期动态生成的测试名，遇到动态字符串时应保持保守处理。
