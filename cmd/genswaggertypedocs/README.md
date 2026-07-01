# genswaggertypedocs 目录说明

`cmd/genswaggertypedocs/` 根据 Go API 类型注释生成 Swagger/OpenAPI 文档函数，或验证类型是否具备必要文档。它把 `types.go` 中的包、类型和字段注释转换为运行时可消费的文档函数。

## 关键文件

- `swagger_type_docs.go`：程序入口，支持 `-s` 指定类型源码、`-f` 指定输出位置、`-v` 执行文档完整性验证。

## 与其他模块的关系

该工具依赖 `k8s.io/apimachinery/pkg/runtime` 的文档解析与输出函数，主要作用于 `staging/src/k8s.io/api/` 等 API 类型定义。生成的文档会进入 OpenAPI、kubectl explain 和用户 API 参考资料链路。

## 开发与测试注意事项

修改 API 类型或字段注释时，应保持注释清晰、稳定并覆盖所有公开类型。验证模式失败通常表示新增类型缺少文档，应补充源码注释，而不是跳过检查。输出到文件时注意不要覆盖非生成文件。
