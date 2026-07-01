# generated 目录说明

`pkg/generated` 是 Kubernetes 仓库内生成代码的集中放置位置之一，目前主要包含 OpenAPI 相关生成结果和生成入口。

## 关键子目录/源码入口

- `openapi/zz_generated.openapi.go`：生成的 Kubernetes OpenAPI 定义。
- `openapi/cmd/models-schema/`：生成或处理模型 schema 的命令入口。
- `openapi/openapi_test.go`：校验生成的 OpenAPI 内容。

## 与 cmd/staging/test 的关系

- `cmd/kube-apiserver` 暴露的 OpenAPI 文档最终依赖这里的生成产物。
- OpenAPI 生成会结合 `pkg/apis`、`staging/src/k8s.io/api` 和 apimachinery 中的类型信息。
- 测试包含本目录单元测试，也可能被 API 兼容性和集成测试间接覆盖。

## 开发/测试注意事项

- 不要手工编辑 `zz_generated.*` 文件；应运行仓库提供的生成脚本。
- API 类型、注释、tag 或校验规则变化后，可能需要重新生成 OpenAPI。
- 修改生成逻辑时，应运行 OpenAPI 相关测试并检查生成 diff 是否只包含预期变化。
