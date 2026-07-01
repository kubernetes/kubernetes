# fieldnamedocscheck 目录说明

`cmd/fieldnamedocscheck/` 提供 API 字段文档检查工具。它读取 Go 类型源码中的结构体和字段注释，检查反引号包裹的字段名是否与 JSON tag 使用的字段名一致，避免 API 文档中出现大小写或命名错误。

## 关键文件

- `field_name_docs_check.go`：程序入口和检查逻辑，通过 `-s`/`--type-src` 指定 API 类型源码文件。

## 与其他模块的关系

该工具依赖 `k8s.io/apimachinery/pkg/runtime` 的文档解析能力，主要服务于 `staging/src/k8s.io/api/` 和相关 API 类型包。检查结果会影响生成的 OpenAPI、kubectl explain 输出以及用户可见 API 文档质量。

## 开发与测试注意事项

修改 API 类型注释时应保持反引号中的字段名与 JSON 字段名一致，例如文档中引用 `apiVersion` 而不是 Go 字段名 `APIVersion`。本工具通常由验证脚本调用，也可用 `go run ./cmd/fieldnamedocscheck -s <types.go>` 针对单个文件排查。
