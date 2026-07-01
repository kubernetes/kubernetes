# apis 目录说明

`pkg/apis` 保存 Kubernetes API 组的内部版本类型、注册、默认值、转换、校验、fuzzer 和安装逻辑，是 kube-apiserver 内存中 API 对象语义的主要定义位置。

## 关键子目录/源码入口

- `core/`、`apps/`、`batch/`、`rbac/`、`networking/`、`storage/` 等：各 API 组的内部类型、注册和校验实现。
- 每个 API 组下的 `v1`、`v1beta1`、`v1alpha*`：对应外部版本的默认值、转换、注册和生成文件。
- `install/`：将 API 组安装到 scheme，供 apiserver 和测试使用。
- `validation/`、`fuzzer/`、`latest/`：分别承载校验、随机对象生成和版本选择等辅助能力。

## 与 cmd/staging/test 的关系

- `cmd/kube-apiserver` 最终通过 control plane 和 registry 代码使用这里的类型注册、转换和校验。
- 外部 Go API 类型位于 `staging/src/k8s.io/api`，本目录维护 Kubernetes 仓库内的内部表示和版本间语义。
- `test/integration`、API 兼容性测试和包内单元测试会验证这里的默认值、转换、校验和生成代码。

## 开发/测试注意事项

- API 变更通常需要同步更新内部类型、外部版本、转换、默认值、校验、openapi 和生成文件。
- 不要手写 `zz_generated.*` 文件；应使用仓库生成脚本。
- 修改已发布 API 时必须关注兼容性、存储版本和降级路径，并运行相关 API 组测试。
