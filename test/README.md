# test 目录源码分析

`test/` 是 Kubernetes 非单元测试和测试基础设施的集中目录。单元测试通常与源码同目录，位于 `pkg/`、`cmd/`、`staging/` 等包内；`test/` 则覆盖集成测试、端到端测试、CLI 测试、测试镜像、conformance、kubemark、metrics 回归和多种验证工具。

顶层 `OWNERS` 由 sig/testing 维护，很多子目录还有各自 SIG 的 OWNERS。修改测试时需要同时关注测试类型、所有权、运行成本和 CI 触发方式。

## 测试分层

```text
源码同目录 *_test.go
  -> 单元测试，速度快，直接验证包内逻辑

test/integration/
  -> 启动 etcd、apiserver、scheduler、controller 等进程内组件

test/cmd/
  -> Bash 驱动的 kubectl 和组件 CLI 集成测试

test/e2e*/*
  -> 面向真实集群或节点的 Ginkgo 端到端测试

test/images/、test/kubemark/、test/conformance/
  -> 测试镜像、规模测试和发布一致性验证
```

## 关键子目录

`test/e2e/` 是集群级 Ginkgo E2E 套件，包含大量按 SIG 划分的场景，也是 conformance 测试的主要来源。入口包括 `test/e2e/e2e_test.go` 和 `test/e2e/e2e.go`，共享框架在 `test/e2e/framework/`。

`test/e2e_node/` 专注 kubelet 和节点行为，通常在节点环境中运行，覆盖容器运行时、Pod 生命周期、资源隔离、日志、探针等场景。

`test/e2e_kubeadm/` 覆盖 kubeadm init、join、upgrade 等集群生命周期场景。

`test/e2e_dra/` 覆盖 Dynamic Resource Allocation 的端到端和升级降级场景，常结合本地集群启动工具运行。

`test/integration/` 是 Go 集成测试集中地，通常会启动真实 etcd 和进程内 apiserver。它直接 import `cmd/kube-apiserver/app`、`pkg/controlplane`、`pkg/scheduler`、`pkg/controller` 等源码包，是验证控制面行为的重要层级。

`test/cmd/` 是 Bash 驱动的 CLI 测试，主要验证 `kubectl` 和核心组件命令在轻量集群环境中的行为。新增脚本通常需要在 `legacy-script.sh` 中注册。

`test/images/` 保存 E2E 测试镜像源码。`agnhost` 是最重要的通用测试镜像，很多新测试应优先扩展 agnhost 子命令，而不是新增独立镜像。

`test/kubemark/` 用 hollow-node 模拟大规模节点，用于控制面规模和性能测试。

`test/conformance/` 保存 conformance 测试列表和黄金数据。修改 conformance 集合通常需要 sig-architecture 关注。

`test/instrumentation/` 维护 stable metrics 回归数据，防止指标名、标签和稳定性级别意外变化。

`test/compatibility_lifecycle/` 校验 feature gate 生命周期和兼容性数据。

`test/fixtures/` 和 `test/e2e/testing-manifests/` 保存嵌入式测试数据，常通过 `//go:embed` 和测试框架 helper 读取。

`test/utils/` 保存跨测试套件共享的工具，例如镜像 manifest、ktesting、本地集群封装等。

`test/typecheck/` 用于快速跨平台类型检查，比完整 cross-build 更轻量。

`test/fuzz/` 保存 fuzz 测试和语料，覆盖 YAML、JSON、CBOR 等序列化路径。

`test/soak/` 保存长时间 soak 测试，用于观察稳定性和资源泄漏。

## 与源码目录的关系

`test/integration/` 与 `pkg/` 的关系最紧密。它会直接启动 apiserver、scheduler、controller-manager 或调用内部 helper，用来验证真实控制面状态机。

`test/cmd/` 与 `cmd/` 关系最紧密。它会构建并运行 `kubectl`、`kube-apiserver`、`kube-controller-manager` 等二进制，验证命令行层和用户可见行为。

`test/e2e/` 主要以黑盒方式通过 Kubernetes API 访问真实集群，通常依赖 `staging/src/k8s.io/client-go`、`apimachinery` 和 e2e framework，而不直接依赖大量 `pkg/` 内部实现。

`test/images/` 与 E2E 强绑定。测试代码通过 `test/utils/image/manifest.go` 引用镜像名和 tag，镜像源码和版本变更需要与该 manifest 同步。

`staging/` 模块在测试中通常以 `k8s.io/client-go`、`k8s.io/apimachinery`、`k8s.io/apiserver`、`k8s.io/component-base` 等模块名被 import；根 `go.mod` 通过 replace 指向本仓库的 `staging/src/k8s.io/*`。

## 常用运行方式

集成测试：

```bash
make test-integration
make test-integration WHAT=./test/integration/scheduler/...
make test-integration WHAT=./test/integration/pods KUBE_TEST_ARGS='-run ^TestName$'
```

CLI 测试：

```bash
make test-cmd
make test-cmd WHAT='deployment impersonation'
```

节点 E2E：

```bash
make test-e2e-node FOCUS=Kubelet
```

集群 E2E 通常先构建 `e2e.test`，再用 ginkgo 脚本指向已有集群：

```bash
make WHAT=test/e2e/e2e.test
hack/ginkgo-e2e.sh
```

测试镜像构建示例：

```bash
make all WHAT=agnhost
```

集成测试通常需要 etcd 在 PATH 中。可以使用 `hack/install-etcd.sh` 下载到仓库的 `third_party/` 目录。

## E2E 开发注意事项

新增 E2E 测试应放在对应 SIG 子目录，并使用 `framework.SIGDescribe` 标注所有权。不要只在测试名中手写 `[sig-xxx]`。

新包或子包可能需要通过 `imports.go` 做 blank import，确保 Ginkgo specs 被注册。

E2E 所有权由 `hack/verify-e2e-test-ownership.sh` 校验，本地可运行：

```bash
make verify WHAT=e2e-test-ownership
```

新增 conformance 测试需要更新 `test/conformance/testdata/`，并遵守 conformance 评审要求。

## Integration 开发注意事项

优先复用 `test/integration/framework/` 启动 apiserver 和 etcd，不要在每个测试里重复搭建基础设施。

调度器、controller、控制面相关场景可查看 `test/integration/util/` 中的 helper。

集成测试运行成本高于单元测试。能在包内单元测试覆盖的逻辑，不应无理由上升到集成测试。

性能测试与行为测试应分离。调度器性能相关用例通常放在 `test/integration/scheduler_perf/`。

## CLI 测试注意事项

`test/cmd/` 是 Bash 测试。新增测试函数应遵循 `run_*_tests` 命名，并在 `legacy-script.sh` 中按约定 source 和调度。

测试资源是否存在时，应使用已有 helper，例如 `kube::test::if_supports_resource`，避免在 API 不启用或特性门控关闭时产生不稳定失败。

## 测试镜像注意事项

修改 `test/images/<image>/` 下镜像内容时，需要 bump 对应 `VERSION` 文件。

合并和发布镜像后，需要同步 `test/utils/image/manifest.go` 中引用的 tag。

新增镜像前应优先考虑扩展 `agnhost`。独立镜像会增加构建、发布和镜像提升流程的维护成本。

## 常见验证入口

```bash
hack/verify-e2e-images.sh
hack/verify-e2e-test-ownership.sh
hack/update-conformance-yaml.sh
make verify WHAT=typecheck
```

这些验证通常和 `test/` 下数据、镜像引用、所有权和跨平台类型检查相关。修改测试基础设施时，建议先查对应子目录 README 和 OWNERS。

