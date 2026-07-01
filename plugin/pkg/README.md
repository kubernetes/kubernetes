# plugin/pkg 目录说明

`plugin/pkg` 保存 Kubernetes 核心仓库内的服务端插件实现，主要覆盖准入控制、认证和授权相关逻辑。这些代码通常不是独立二进制入口，而是由 API Server、Controller Manager 或测试代码按插件名注册和调用。

## 关键子目录/源码入口

- `admission/`：内置准入插件实现，例如 ServiceAccount、NodeRestriction、ResourceQuota、PodSecurity、LimitRanger、存储和网络相关准入逻辑。各插件通常以 `admission.go` 为入口，并配套 `*_test.go`。
- `auth/`：认证和授权插件代码，包括 bootstrap token 认证、Node authorizer、RBAC authorizer 以及启动时需要的 RBAC bootstrap policy。
- `auth/authorizer/rbac/bootstrappolicy/`：生成和校验默认 ClusterRole、RoleBinding 等引导策略的核心位置，测试数据保存在 `testdata/`。

## 与 Kubernetes 其他模块的关系

这些插件处在 API 请求链路中，依赖 `k8s.io/api`、`k8s.io/apimachinery`、`k8s.io/apiserver`、`k8s.io/client-go` 以及 `pkg/` 下的内部 API 辅助逻辑。准入插件影响对象创建、更新和删除；认证/授权插件影响请求身份识别和权限判定，因此与 API Server 启动配置、feature gate、RBAC 默认策略、节点安全模型和控制器行为密切相关。

## 开发/测试注意事项

修改插件时应优先补充同目录单元测试，并关注 admission chain、authorization decision 和 bootstrap policy 的兼容性。涉及 API 类型、默认值或转换代码时，确认是否需要运行代码生成。影响默认权限或安全边界的改动需要同时检查集成测试、e2e 测试以及升级兼容性，避免破坏已有集群的默认角色和准入行为。
