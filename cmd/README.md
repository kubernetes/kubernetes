# cmd 目录源码分析

`cmd/` 是 Kubernetes 官方二进制的入口层。这里的代码负责组装命令行、flags、日志、metrics、认证插件注册和组件启动流程；大多数业务逻辑会继续委托到 `pkg/` 或 `staging/src/k8s.io/*`。

换句话说，`cmd/` 主要回答“这个二进制如何启动”，而不是“调度、存储、网络或 kubelet 的核心算法如何实现”。

## 目录职责

`cmd/` 下的目录大致分三类。

核心集群组件包括 `kube-apiserver/`、`kube-controller-manager/`、`kube-scheduler/`、`kubelet/`、`kube-proxy/`、`kubectl/`、`kubectl-convert/`、`kubeadm/`、`cloud-controller-manager/` 和 `kubemark/`。

文档与代码生成工具包括 `gendocs/`、`genman/`、`genyaml/`、`genkubedocs/`、`genswaggertypedocs/`、`genfeaturegates/` 和 `gotemplate/`。

仓库治理和 CI 工具包括 `import-boss/`、`importverifier/`、`preferredimports/`、`dependencycheck/`、`dependencyverifier/`、`clicheck/`、`fieldnamedocscheck/` 和 `prune-junit-xml/`。

顶层 `OWNERS` 对 `cmd/` 开启了独立所有权，修改入口层代码通常需要该目录维护者参与审查。

## 常见入口模式

大多数组件采用“薄 main + app 包”的结构。

```text
cmd/<component>/<main>.go
    -> cmd/<component>/app.New<Component>Command()
    -> k8s.io/component-base/cli.Run(...)
    -> pkg/ 或 staging 中的真实实现
```

这些 `main` 文件通常只做三件事：引入 side-effect 注册包，创建 Cobra 命令，执行并把退出码传给 `os.Exit`。

典型 side-effect import 包括：

- `k8s.io/component-base/logs/json/register`：注册 JSON 日志格式。
- `k8s.io/component-base/metrics/prometheus/clientgo`：注册 client-go metrics。
- `k8s.io/component-base/metrics/prometheus/version`：注册版本指标。
- `k8s.io/client-go/plugin/pkg/client/auth`：注册 kubectl 可用的云厂商认证插件。
- `time/tzdata`：为依赖时区数据库的组件提供内置数据。

## 核心二进制导航

`kube-apiserver` 的入口在 `cmd/kube-apiserver/apiserver.go`，配置和启动逻辑在 `cmd/kube-apiserver/app/`。它会进一步组装三层 apiserver：`staging/src/k8s.io/kube-aggregator` 处理聚合 API，`staging/src/k8s.io/apiextensions-apiserver` 处理 CRD，`pkg/controlplane` 和 `pkg/kubeapiserver` 处理 Kubernetes 核心 API、认证、授权、admission 和 OpenAPI。

`kube-controller-manager` 的入口在 `cmd/kube-controller-manager/controller-manager.go`。`cmd/kube-controller-manager/app/` 负责 controller 注册、flags、leader election 和启动编排，实际控制器实现位于 `pkg/controller/*`。`cmd/kube-controller-manager/names/` 保存 controller 名称常量。

`kube-scheduler` 的入口在 `cmd/kube-scheduler/scheduler.go`。命令层负责加载配置和创建调度器实例，调度循环、framework runtime 和内置插件位于 `pkg/scheduler/`。

`kubelet` 的入口在 `cmd/kubelet/kubelet.go`。`cmd/kubelet/app/` 处理 kubelet flags、配置文件、特性门控和依赖注入，节点生命周期、Pod worker、CRI、PLEG、volume manager、eviction 和资源管理位于 `pkg/kubelet/`。

`kube-proxy` 的入口在 `cmd/kube-proxy/proxy.go`。网络代理逻辑位于 `pkg/proxy/`，包括 iptables、IPVS、nftables 和 Windows proxier。

`kubectl` 是一个特例。入口在 `cmd/kubectl/kubectl.go`，但命令实现主要在 `staging/src/k8s.io/kubectl/pkg/cmd/`。因此新增 kubectl 子命令或修改用户可见行为时，通常不应只看 `cmd/kubectl/`。

`kubeadm` 也是特例。它的逻辑大量保留在 `cmd/kubeadm/app/`，包括 `app/cmd/` 的 Cobra 命令树、`app/phases/` 的 init/join/upgrade 阶段和 `app/apis/` 的配置 API。

`cloud-controller-manager` 的入口委托给 `staging/src/k8s.io/cloud-provider/app`。云厂商扩展不应直接 vendor 当前目录，而应基于 `k8s.io/cloud-provider` 构建。

## 启动流水线

组件常见启动流程是：

```text
NewOptions
  -> AddFlags
  -> Complete
  -> Validate
  -> Run
```

`Complete` 通常把命令行选项补全为运行所需的完整配置，`Validate` 检查非法组合，`Run` 开始创建 informer、client、server、controller 或 scheduler。

`kube-apiserver` 会在这个流程中完成认证授权配置、admission 插件注册、API group 安装、OpenAPI 配置和 shutdown handler 组装。

`kube-controller-manager` 会把各类 controller descriptor 统一注册，再依据 flags、feature gates、leader election 和资源可用性决定启动哪些 controller。

`kubelet` 的 flags 处理更特殊，历史兼容和配置优先级较复杂，因此 `cmd/kubelet/app/server.go` 中包含更多手动解析和合并逻辑。

## 与 pkg 和 staging 的边界

`cmd/` 可以 import `pkg/` 和 `staging/src/k8s.io/*`，但 `pkg/` 不应反向依赖 `cmd/`。这种方向保证组件逻辑可被测试和复用，不需要通过真实二进制入口才能运行。

`pkg/` 保存 Kubernetes 主仓库内的核心实现，例如 scheduler、kubelet、proxy、controller、registry 和 controlplane。

`staging/` 保存可独立发布的模块，例如 `client-go`、`apimachinery`、`apiserver`、`component-base`、`kubectl`、`cloud-provider` 等。

开发时如果只是在修改业务逻辑，通常应先定位到 `pkg/` 或 `staging/`；只有启动参数、命令结构、组件 wiring 或二进制特有注册逻辑才应落在 `cmd/`。

## 构建和调试

常用构建命令：

```bash
make all WHAT=cmd/kube-apiserver
make all WHAT=cmd/kubelet
make all WHAT=cmd/kubectl
make all WHAT=cmd/kube-scheduler DBG=1
```

构建产物默认位于 `_output/local/bin/`。`DBG=1` 会保留更适合调试的未裁剪二进制。

本地多组件集群常通过 `hack/local-up-cluster.sh` 启动。集成测试中更常见的方式是直接使用 `cmd/kube-apiserver/app/testing/`、`cmd/kube-controller-manager/app/testing/` 或 `cmd/kube-scheduler/app/testing/` 的测试 server，而不是手工启动完整二进制。

## 开发注意事项

新增或修改组件 flags 时，需要同时考虑配置文件 API、默认值、验证逻辑、文档生成和兼容性。很多 flag 已经被迁移到 ComponentConfig，新增 flag 应谨慎。

新增 controller 时，不只要实现 `pkg/controller/...`，还要在 `cmd/kube-controller-manager/app/` 中注册 descriptor，并在 `names/` 中维护名称常量。

新增 admission 插件时，实际实现位于 `plugin/pkg/admission/`，但注册顺序和默认启用策略在 `pkg/kubeapiserver/options/plugins.go`，最终由 `cmd/kube-apiserver` 启动时接入。

修改 kubectl 行为时，优先查看 `staging/src/k8s.io/kubectl/pkg/cmd/` 和 `staging/src/k8s.io/cli-runtime/`。

修改 import 约束、依赖策略或生成工具时，注意对应工具可能本身就在 `cmd/` 下，CI 会通过 `hack/verify-import-boss.sh`、`hack/verify-imports.sh` 等脚本执行它们。

