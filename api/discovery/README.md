# discovery 目录说明

`api/discovery/` 保存 Kubernetes API Discovery 端点的静态 JSON 快照。这里记录的是 kube-apiserver 对外暴露的资源发现结果，包括 legacy core API 的 `/api`、聚合 API 的 `/apis`，以及各 API group/version 下的资源、版本和能力信息。

## 关键文件

- `api.json`、`api__v1.json`：对应 core API 的发现结果。
- `apis.json`：对应聚合 API group 列表。
- `apis__<group>__<version>.json`：对应某个 API group/version 的资源发现结果，文件名中的 `__` 表示 URL 路径分隔符。
- `aggregated_v2.json`：聚合后的 discovery v2 数据快照。

## 与其他模块的关系

这些文件由真实 kube-apiserver 输出生成，反映 `staging/src/k8s.io/api/`、`pkg/apis/`、API 安装逻辑和 feature gate 共同决定的对外 API 面。客户端、文档生成、兼容性验证和部分测试会把它们作为 API 合约基线使用。

## 开发与测试注意事项

不要手动编辑这些 JSON 快照。修改 API 类型、REST 存储、资源版本或 feature gate 暴露行为后，应运行 `hack/update-openapi-spec.sh` 更新本目录，并用 `hack/verify-openapi-spec.sh` 验证生成结果。新增资源时要确认 discovery、OpenAPI 和相关文档变更一致。
