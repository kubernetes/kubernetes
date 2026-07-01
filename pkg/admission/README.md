# admission 目录说明

`pkg/admission` 放置 Kubernetes 专有的准入控制辅助代码，主要为 kube-apiserver 加载本地准入 Webhook 和 Admission Policy 清单时接入 Kubernetes API scheme、默认值和校验逻辑。

## 关键子目录/源码入口

- `plugin/webhook/manifest/loader/`：加载 `ValidatingWebhookConfiguration` 和 `MutatingWebhookConfiguration` 清单，复用 `staging/src/k8s.io/apiserver` 中的通用加载循环。
- `plugin/policy/manifest/loader/`：加载 `ValidatingAdmissionPolicy`、`MutatingAdmissionPolicy` 及其 binding 清单。

## 与 cmd/staging/test 的关系

- `cmd/kube-apiserver` 通过 apiserver 配置路径间接使用这些加载器。
- 通用准入加载能力位于 `staging/src/k8s.io/apiserver`，本目录负责补齐 Kubernetes 内部 API 类型注册、转换和校验。
- 单元测试随源码放在对应 loader 目录，集成行为可能由 apiserver 相关测试覆盖。

## 开发/测试注意事项

- 修改清单加载行为时，要保持文件处理顺序、hash 计算和错误返回的确定性。
- 新增支持的资源类型时，需要同步安装 scheme、默认值和 validation。
- 建议运行相关 loader 包测试，并根据影响范围补充 kube-apiserver 准入配置测试。
