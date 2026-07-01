# api 目录源码分析

`api/` 保存 Kubernetes API 的静态规范产物和 API 规则基线。这里不是 Go API 类型定义所在的位置；真正对外发布的 API 类型在 `staging/src/k8s.io/api/`，运行时使用的 internal 类型在 `pkg/apis/`。

这个目录更像 API 合约的快照层：它把 kube-apiserver 暴露出来的 OpenAPI、Discovery 信息以及已知 API 规则例外固化到仓库中，供 CI、文档、客户端工具和回归测试使用。

## 目录结构

```text
api/
├── api-rules/
├── discovery/
├── openapi-spec/
└── OWNERS
```

`api-rules/` 记录 API 约定检查中已经被接受的历史例外。`openapi-gen` 在代码生成时会检查 API 类型命名、list 语义、字段标记等规则，新增违规通常会导致验证失败。这里的 `violation_exceptions.list` 以及各 staging 子项目的例外清单应被当作源码审查，而不是普通生成物。

`discovery/` 保存 kube-apiserver Discovery 端点的 JSON 快照，包括 legacy core API 的 `/api`、聚合 API 的 `/apis`，以及各 group/version 下的资源列表。文件名把 URL 路径中的 `/` 转换为 `__`，例如 `apis/apps/v1` 对应 `apis__apps__v1.json`。

`openapi-spec/` 保存 OpenAPI 快照。`swagger.json` 是 OpenAPI v2 全量单文件，`v3/` 目录按路径拆分 OpenAPI v3 文档，例如 `api__v1_openapi.json`、`apis__apps__v1_openapi.json`。该目录下的 README 还解释了 Kubernetes 特有的 OpenAPI 扩展字段，例如 `x-kubernetes-list-type`、`x-kubernetes-patch-strategy`、`x-kubernetes-group-version-kind`。

## 与 API 源码的关系

API 类型、生成代码和本目录快照之间的关系可以概括为：

```text
staging/src/k8s.io/api/**/types.go
pkg/apis/**/*
        |
        | hack/update-codegen.sh
        v
pkg/generated/openapi/zz_generated.openapi.go
        |
        | kube-apiserver 运行时暴露 /openapi 与 /apis
        | hack/update-openapi-spec.sh 抓取
        v
api/openapi-spec/*
api/discovery/*
```

`staging/src/k8s.io/api/` 中的 `types.go` 是 external API 类型，例如 Pod、Deployment、Service 等。包级 `doc.go` 常带有 `+k8s:openapi-gen=true` 标记，用来驱动 OpenAPI 生成。

`pkg/apis/` 中是 apiserver 内部使用的 internal 类型、默认值、转换和验证逻辑。很多 internal 包通过注释标记连接到 external API，生成 validation、conversion、defaulter 等代码。

`pkg/generated/openapi/zz_generated.openapi.go` 是从类型源码生成的 Go 版 OpenAPI 定义。kube-apiserver 启动时把这些定义挂接到 OpenAPI 服务，再由 `hack/update-openapi-spec.sh` 启动真实 apiserver 并抓取 HTTP 端点，最终更新本目录中的 JSON 快照。

## 关键生成和验证入口

`hack/update-codegen.sh` 负责运行 OpenAPI、validation、conversion、defaulter 等代码生成流程。它会更新 Go 生成代码，并在 API 规则检查阶段使用 `api/api-rules/` 中的例外清单。

`hack/update-openapi-spec.sh` 会构建并启动 kube-apiserver，开启完整 API 和特性门控后抓取 OpenAPI v2/v3 与 Discovery 文档。该脚本依赖 etcd、jq 和已编译的 apiserver。

`hack/verify-openapi-spec.sh` 用于确认 `api/openapi-spec/` 与 `api/discovery/` 是否和当前源码生成的 apiserver 输出一致。CI 中若该验证失败，通常说明 API 类型、REST 路由或 feature gate 改动后没有同步更新静态快照。

`hack/verify-openapi-docs-urls.sh` 检查 OpenAPI 描述中引用的文档 URL。这个验证不一定在默认 CI 中运行，但对维护 API 文档链接很有用。

## 下游消费者

本目录的 JSON 文件不仅是文档产物，也被多个测试和工具消费。

`staging/src/k8s.io/apimachinery/pkg/util/managedfields/` 的测试会读取 OpenAPI 信息构造 managed fields 的类型转换器。

`staging/src/k8s.io/client-go/openapi/openapitest/` 使用与 `api/openapi-spec/v3/` 命名方式一致的测试客户端和精简数据。

`staging/src/k8s.io/apiextensions-apiserver/pkg/apiserver/testdata/` 中的部分测试数据来源于 `swagger.json` 裁剪结果，用于 CRD 相关 schema 行为测试。

文档生成、API 兼容性检查和 Server-Side Apply 行为验证都会间接受到这些快照的影响。

## 修改指南

修改 `staging/src/k8s.io/api/**`、`pkg/apis/**` 或与 API schema 相关的注释后，通常需要运行 `hack/update-codegen.sh`。

新增 API group/version、REST 路径、子资源，或改变 kube-apiserver 对外暴露的资源后，通常需要运行 `hack/update-openapi-spec.sh`。

如果 API 规则检查新增违规，应优先修复 API 设计。只有在确认为历史兼容或经过 API review 接受时，才更新 `api/api-rules/*.list`，并在评审中明确说明原因。

不要手动编辑 `api/openapi-spec/` 和 `api/discovery/` 下的大型 JSON 文件。它们应由脚本生成，以保证格式、排序和版本字段一致。

## 常见误区

`api/discovery/` 不是 EndpointSlice 的 Go API 包。EndpointSlice 类型位于 `staging/src/k8s.io/api/discovery/`。

`api/openapi-spec/swagger.json` 和 `api/openapi-spec/v3/` 不是简单的一份文件和拆分文件关系。v2 是单文件全量规范，v3 按路径拆分，还包含版本、日志和 OIDC 等非资源端点。

`api/api-rules/` 不是可随意刷新的生成目录。它记录的是 API 规则债务基线，目标应是逐步减少例外，而不是为新违规扩容。

