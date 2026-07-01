# quota 目录说明

`pkg/quota` 实现 Kubernetes ResourceQuota 相关的配额评估器注册、核心资源用量计算和 admission 更新过滤逻辑。

关键入口和子目录：
- `v1/evaluator/core/` 保存 Pod、Service、PVC、ResourceClaim 等核心资源的配额评估器。
- `v1/evaluator/core/registry.go` 注册核心资源评估器集合。
- `v1/install/` 将配额评估器安装到 apiserver 使用的注册表，并提供更新过滤逻辑。

与 `cmd`、`staging`、`test` 的关系：
- apiserver 命令入口通过控制面启动链路装配 admission 和资源存储，间接使用本目录的配额评估能力。
- 本目录依赖 staging 中的 API、apimachinery 和 admission 相关接口来计算资源用量。
- 单元测试位于各 evaluator 和 install 子目录；配额 admission 的集成行为还会被 apiserver 和 e2e 测试覆盖。

开发和测试注意事项：
- 新增或调整资源计量时要明确资源名、scope、对象更新语义和零值处理。
- 配额计算会影响准入结果，修改时应覆盖创建、更新、删除和状态变化场景。
- 保持 evaluator 只负责用量计算，不要在此处引入存储层或命令行启动逻辑。
