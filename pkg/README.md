# pkg 目录源码分析

`pkg/` 是 Kubernetes 主仓库内的核心业务实现层。`cmd/` 负责启动和命令行 wiring，`staging/` 负责可独立发布的公共模块，而 `pkg/` 保存 Kubernetes 集群本身的主要逻辑：控制面、调度、节点、网络、控制器、registry、内部 API、插件集成和各类工具函数。

这里的代码通常不会作为独立 `k8s.io/*` 模块对外发布，因此它可以承载 Kubernetes 主仓库特有的实现细节。

## 顶层结构概览

`pkg/apis/` 保存 apiserver 内部使用的 internal API 类型、默认值、转换、验证和生成代码。对外发布的 versioned API 类型在 `staging/src/k8s.io/api/`。

`pkg/api/` 保存跨 API 版本的辅助逻辑、legacy scheme、Pod/Service 工具和兼容性测试。

`pkg/registry/` 是 kube-apiserver 的 REST 存储层。每个资源通常在这里定义 strategy、storage、子资源处理和与 etcd store 的连接。

`pkg/controlplane/` 组装 Kubernetes 核心控制面，包括 API group 安装、server chain、内嵌 controller、Endpoints/Lease reconciler 等。

`pkg/kubeapiserver/` 保存 kube-apiserver 专有的认证、授权、admission、选项和配置逻辑。通用 apiserver 逻辑在 `staging/src/k8s.io/apiserver/`。

`pkg/controller/` 是 kube-controller-manager 内置 controller 的主要实现位置。

`pkg/scheduler/` 是 kube-scheduler 的完整实现，包括调度循环、queue、framework runtime、profile 和内置插件。

`pkg/kubelet/` 是节点 agent 的主体，包含 Pod 生命周期、CRI runtime、PLEG、volume manager、eviction、资源管理、HTTP server 和状态上报。

`pkg/proxy/` 是 kube-proxy 的网络代理实现，包含 iptables、IPVS、nftables、Windows proxier 和 service/endpointslice 变更跟踪。

`pkg/volume/` 保存内置 volume 插件和挂载逻辑，包括 CSI、emptyDir、hostPath、projected、secret、configMap 等。

其他基础目录包括 `features/`、`generated/`、`admission/`、`auth/`、`quota/`、`serviceaccount/`、`credentialprovider/`、`security/`、`routes/`、`util/`、`windows/` 等。

## 与 cmd 和 staging 的边界

整体依赖方向应保持为：

```text
cmd/  ->  pkg/  ->  staging/src/k8s.io/*
```

`cmd/` 创建命令、解析 flags、加载配置并启动组件。`pkg/` 实现组件行为。`staging/` 提供可复用基础库和公开模块，例如 `client-go`、`apimachinery`、`apiserver`、`component-base`、`kubectl` 和各组件配置 API。

`pkg/` 可以依赖 staging 模块，但不应依赖 `cmd/`。仓库通过 `pkg/.import-restrictions` 和相关 verify 脚本维护这条边界。

`staging/` 一般不能依赖 `pkg/`，因为 staging 会被发布到独立仓库。少数历史例外由 `staging/publishing/import-restrictions.yaml` 管理，新增例外需要非常谨慎。

## 控制面源码路径

kube-apiserver 的主要调用链是：

```text
cmd/kube-apiserver/app
  -> pkg/controlplane
  -> pkg/controlplane/apiserver
  -> pkg/kubeapiserver
  -> pkg/registry/*/rest
  -> pkg/apis/*
```

`pkg/controlplane/instance.go` 负责创建控制面实例并安装 API。`pkg/controlplane/apiserver/config.go` 和 `server.go` 负责 apiserver 配置与 server chain。`pkg/registry/core/rest/storage_core.go` 展示了 core API 资源如何注册为 REST storage。

`pkg/kubeapiserver/options/` 保存 kube-apiserver 专有选项，包括认证、授权、admission、API enablement 等。通用 server、storage、request handling 和 admission 框架在 `staging/src/k8s.io/apiserver/`。

控制面内嵌 controller 位于 `pkg/controlplane/controller/`，例如 Kubernetes 默认 Service、ServiceCIDR、CRD registration 和 leader election 相关控制器。它们不同于 kube-controller-manager 中的业务 controller。

## Controller Manager 源码路径

`cmd/kube-controller-manager/app/` 负责注册和启动 controller，实际 controller 代码在 `pkg/controller/`。

典型 controller 包包括：

- `pkg/controller/deployment/`：Deployment 控制器。
- `pkg/controller/replicaset/`：ReplicaSet 控制器。
- `pkg/controller/statefulset/`：StatefulSet 控制器。
- `pkg/controller/job/`：Job 控制器。
- `pkg/controller/nodeipam/`：Node IPAM。
- `pkg/controller/endpoint/` 和 `pkg/controller/endpointslice/`：Endpoint 与 EndpointSlice。
- `pkg/controller/resourceclaim/`：DRA 相关资源控制。

这些 controller 通常围绕 informer、workqueue、sync handler 和 status update 展开。新增 controller 需要同时考虑 `cmd/kube-controller-manager/app/` 的注册、controller 名称、feature gate、权限和测试。

## Scheduler 源码路径

`pkg/scheduler/scheduler.go` 定义 scheduler 主体，`pkg/scheduler/schedule_one.go` 是单个 Pod 调度路径的核心。调度框架 runtime 位于 `pkg/scheduler/framework/runtime/`，内置插件位于 `pkg/scheduler/framework/plugins/`。

调度流程通常包含队列取 Pod、PreFilter、Filter、PostFilter、Score、Reserve、Permit、PreBind、Bind、PostBind 等阶段。插件接口对外定义在 `staging/src/k8s.io/kube-scheduler/framework/`，内置实现保留在 `pkg/scheduler/framework/plugins/`。

调度器内部 ComponentConfig 位于 `pkg/scheduler/apis/config/`，对外 versioned 配置类型在 `staging/src/k8s.io/kube-scheduler/config/`。

## Kubelet 源码路径

`pkg/kubelet/kubelet.go` 定义 kubelet 主结构体。`pkg/kubelet/pod_workers.go` 负责串行化和调度 Pod 同步工作。`pkg/kubelet/kuberuntime/kuberuntime_manager.go` 对接 CRI runtime。`pkg/kubelet/pleg/` 负责 Pod 生命周期事件。

节点资源和生命周期相关逻辑分布在：

- `pkg/kubelet/cm/`：CPU、内存、拓扑和 cgroup 资源管理。
- `pkg/kubelet/eviction/`：节点压力驱逐。
- `pkg/kubelet/volumemanager/`：volume attach/mount 生命周期。
- `pkg/kubelet/server/`：kubelet HTTP server、stats、exec、logs 等接口。
- `pkg/kubelet/status/`：Pod 和 Node 状态生成与上报。

kubelet 配置 internal 类型在 `pkg/kubelet/apis/config/`，对外配置类型在 `staging/src/k8s.io/kubelet/config/`。

## 网络与 kube-proxy

`pkg/proxy/` 实现三层 Service 代理。公共逻辑包括 service/endpointslice 变更跟踪、node 信息监听和 proxier 接口。具体平台和后端实现位于：

- `pkg/proxy/iptables/`
- `pkg/proxy/ipvs/`
- `pkg/proxy/nftables/`
- `pkg/proxy/winkernel/`

Service 的 apiserver 存储和 ClusterIP 分配逻辑在 `pkg/registry/core/service/`，EndpointSlice controller 在 `pkg/controller/endpointslice/`，NetworkPolicy、Ingress、IPAddress、ServiceCIDR 等 registry 位于 `pkg/registry/networking/`。

## API、Registry 与生成代码

Kubernetes API 在主仓库内有 internal 和 external 两套形态。`pkg/apis/` 是 apiserver 内部形态，`staging/src/k8s.io/api/` 是对外版本化形态。二者通过 conversion、defaults、validation 和 scheme 连接。

`pkg/registry/` 将 API 类型注册为 REST storage。一个资源通常包含 strategy、storage、REST 子资源、状态更新、删除校验和表格打印等逻辑。

`pkg/generated/` 保存部分集中生成代码，尤其是 `pkg/generated/openapi/zz_generated.openapi.go`。更多 `zz_generated.*.go` 分散在各 API 包中，由 `hack/update-codegen.sh` 维护。

## 测试与开发

单元测试与源码同目录，通常直接运行：

```bash
go test k8s.io/kubernetes/pkg/scheduler/...
go test k8s.io/kubernetes/pkg/kubelet/...
go test k8s.io/kubernetes/pkg/proxy/...
```

集成测试位于 `test/integration/`，会直接 import `pkg/controlplane`、`pkg/scheduler`、`pkg/controller` 等包并启动内嵌组件。

修改 API 类型时，通常需要同步 `pkg/apis/`、`staging/src/k8s.io/api/`，并运行 `hack/update-codegen.sh`。修改对外 OpenAPI 或 Discovery 行为时，还可能需要运行 `hack/update-openapi-spec.sh`。

新增 feature gate 时，Kubernetes 专有 gate 通常在 `pkg/features/kube_features.go` 注册。如果 gate 影响通用 apiserver 或 staging 模块，需要确认 staging 侧是否也需要同步。

新增跨包 import 时，先检查 `.import-restrictions`。违反 import 方向的依赖会在 `hack/verify-import-boss.sh` 或 `hack/verify-imports.sh` 中失败。

## 阅读建议

理解 apiserver，从 `cmd/kube-apiserver/app/` 进入，再沿着 `pkg/controlplane/`、`pkg/kubeapiserver/`、`pkg/registry/` 和 `pkg/apis/` 阅读。

理解调度，从 `cmd/kube-scheduler/app/` 进入，再阅读 `pkg/scheduler/scheduler.go`、`schedule_one.go` 和 `framework/plugins/`。

理解节点，从 `cmd/kubelet/app/` 进入，再阅读 `pkg/kubelet/kubelet.go`、`pod_workers.go`、`kuberuntime/` 和 `volumemanager/`。

理解 Service 网络，从 `cmd/kube-proxy/app/` 进入，再阅读 `pkg/proxy/`、`pkg/controller/endpointslice/` 和 `pkg/registry/core/service/`。

