# plugin 目录源码分析

`plugin/` 保存 Kubernetes 专用的 apiserver 插件实现，主要覆盖认证、授权和 admission。这里的代码会被 kube-apiserver 编译进二进制，并通过 `pkg/kubeapiserver/` 的选项和注册逻辑接入运行时。

需要注意，`staging/src/k8s.io/apiserver/.../plugin/` 是通用 apiserver 插件位置，而顶层 `plugin/` 是 Kubernetes 主仓库特有的插件实现。

## 目录结构

```text
plugin/
└── pkg/
    ├── admission/
    └── auth/
```

`plugin/pkg/auth/` 包含 Kubernetes 专有认证和授权实现，例如 bootstrap token、RBAC、Node authorizer 和默认 bootstrap policy。

`plugin/pkg/admission/` 包含 Kubernetes 专有 admission 插件，例如 NodeRestriction、ServiceAccount、PodSecurity、OwnerReferencesPermissionEnforcement、Certificate 相关插件和若干策略插件。

## 与 kube-apiserver 的接线

插件实现本身位于 `plugin/`，但注册和启用策略不在这里完成。主要接线点包括：

- `pkg/kubeapiserver/options/plugins.go`：注册 admission 插件、维护默认顺序和默认关闭列表。
- `pkg/kubeapiserver/options/authentication.go`：接入 Kubernetes 专用认证选项和 bootstrap token 认证。
- `pkg/kubeapiserver/authorizer/config.go`：组装 RBAC、Node、Webhook 等授权模式。
- `pkg/registry/rbac/rest/storage_rbac.go`：启动时写入 RBAC bootstrap policy。

kube-apiserver 启动时会从 `cmd/kube-apiserver/app/` 进入，经过 `pkg/kubeapiserver/options/` 和 `pkg/controlplane/` 完成插件注册、配置解析和 admission chain 构建。

## 认证与授权插件

`plugin/pkg/auth/authenticator/token/bootstrap/` 实现 bootstrap token 认证。kubeadm join 等流程会依赖 bootstrap token，认证器通过 kube-system 中的 Secret 校验 token 并生成对应用户和组信息。

`plugin/pkg/auth/authorizer/rbac/` 实现 RBAC 授权器。它根据 Role、ClusterRole、RoleBinding、ClusterRoleBinding 判断请求是否被允许。

`plugin/pkg/auth/authorizer/rbac/bootstrappolicy/` 定义 Kubernetes 内置的默认 ClusterRole、Role、Binding 和 controller 权限。这里的改动会直接影响系统组件的默认权限边界，必须结合测试数据和安全评审谨慎处理。

`plugin/pkg/auth/authorizer/node/` 实现 Node authorizer。它限制 kubelet 只能访问与自身节点相关的 Pod、Secret、ConfigMap、PVC 等资源。该 authorizer 与 `plugin/pkg/admission/noderestriction/` 共同构成 Kubernetes 节点身份安全边界。

## Admission 插件

Admission 插件在请求通过认证和授权之后执行，可在对象持久化前进行默认化、校验、拒绝或补充。

关键插件包括：

- `plugin/pkg/admission/noderestriction/`：限制 Node 身份只能修改与自身节点相关的对象和字段。
- `plugin/pkg/admission/serviceaccount/`：处理 ServiceAccount 自动挂载、引用校验和相关默认行为。
- `plugin/pkg/admission/security/podsecurity/`：实现 Pod Security 标准。
- `plugin/pkg/admission/gc/`：实现 OwnerReferences 权限强制，防止用户借 ownerReference 越权控制垃圾回收。
- `plugin/pkg/admission/certificates/`：处理 CSR 审批、签名和 subject 限制等证书相关 admission。
- `plugin/pkg/admission/eventratelimit/`：事件限流 admission，并带有独立配置 API。
- `plugin/pkg/admission/podtolerationrestriction/`：限制 Pod toleration，并带有独立配置 API。

部分 admission 只是 Kubernetes 对通用 apiserver admission 的集成或测试包装。阅读时应区分实现主体是在 `plugin/`，还是在 `staging/src/k8s.io/apiserver/`。

## 插件顺序

Admission 插件顺序由 `pkg/kubeapiserver/options/plugins.go` 中的有序列表维护。顺序是行为的一部分，尤其是和默认值、准入校验、webhook、quota、deny 类插件相关的场景。

新增插件时通常需要同时处理：

1. 在 `plugin/pkg/admission/<name>/` 中实现插件。
2. 在 `pkg/kubeapiserver/options/plugins.go` 中注册插件。
3. 决定是否加入默认启用列表，或放入默认关闭列表。
4. 增加单元测试、集成测试和必要的 e2e 覆盖。
5. 检查插件位置是否应早于 webhook、ResourceQuota 或 AlwaysDeny。

## 与 cluster 目录的关系

`plugin/` 是 apiserver 二进制内的安全策略实现，`cluster/` 是遗留集群部署脚本和 addon 清单。两者处于不同层级，但会在默认集群配置中相互呼应。

例如 GCE 部署脚本会配置 `--authorization-mode=Node,RBAC`，对应 `plugin/pkg/auth/authorizer/node/` 和 `plugin/pkg/auth/authorizer/rbac/`。

GCE 默认 admission 列表会启用 `NodeRestriction` 等插件，对应 `plugin/pkg/admission/noderestriction/`。

`cluster/addons/rbac/` 中的 addon RBAC YAML 是部署层补充权限；`plugin/pkg/auth/authorizer/rbac/bootstrappolicy/` 是 apiserver 启动时写入的内置系统权限。二者不是同一来源。

## 安全开发注意事项

认证、授权和 admission 都是安全边界。修改这些插件时，应优先考虑向后兼容、权限收缩或扩张、升级场景和现有集群中对象的历史状态。

RBAC bootstrap policy 的变更需要同步测试快照。新增权限应说明具体系统组件为什么需要，删除权限应评估升级和降级期间是否会破坏已有控制器。

Node authorizer 和 NodeRestriction 应一起理解。前者控制“能访问什么资源”，后者控制“即使能访问，哪些字段或对象仍不能修改”。

Admission 插件可能影响所有对象写入路径。新增拒绝条件时，需要确认 dry-run、server-side apply、subresource、update/status、delete、connect 等请求类型是否都符合预期。

带配置 API 的插件需要维护 internal/external 类型、默认值、转换、验证和文档。不要只实现 runtime 行为而忽略配置版本化。

## 测试建议

插件单元测试通常与实现同目录，直接运行对应包即可：

```bash
go test k8s.io/kubernetes/plugin/pkg/auth/...
go test k8s.io/kubernetes/plugin/pkg/admission/...
```

涉及 apiserver 链路时，建议补充 `test/integration/` 下的集成测试，验证真实 admission chain、authorization chain 或 RBAC bootstrap 行为。

涉及用户可见安全策略时，通常还需要 e2e 或 conformance 视角的覆盖，避免只测试内部 helper。

