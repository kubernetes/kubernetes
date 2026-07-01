# test/kubemark 目录说明

`test/kubemark` 保存 Kubemark 测试集群的启动、停止、配置和 e2e 执行脚本。Kubemark 使用 hollow node 模拟大量节点，主要用于规模、性能和集群行为验证，而不需要为每个节点创建完整真实机器。

## 关键子目录/源码入口

- `start-kubemark.sh`：创建 Kubemark 集群、构建并上传 hollow-node 镜像、生成资源清单并创建 hollow node 相关资源。
- `stop-kubemark.sh`：清理 Kubemark 集群资源。
- `run-e2e-tests.sh`：配置 Kubemark provider 环境变量，并通过 `hack/ginkgo-e2e.sh` 运行 e2e 测试。
- `configure-kubectl.sh`、`cloud-provider-config.sh`：准备 kubeconfig 和云厂商相关配置。
- 云厂商、资源模板和 skeleton 脚本通常由本目录下的 provider 子目录、`resources/` 以及 `cluster/kubemark/` 配合提供。

## 与 Kubernetes 其他模块的关系

Kubemark 依赖 `cmd/kubemark`、`cluster/images/kubemark`、`cluster/kubemark`、`cluster/` provider 脚本以及标准 e2e framework。它通过真实控制面和模拟节点压测 API Server、调度器、控制器、kubelet/kube-proxy 行为，是规模测试和性能验证链路的一部分。

## 开发/测试注意事项

修改脚本时要兼顾不同云厂商 provider 的环境变量和 shell 兼容性，避免破坏现有 CI 作业。运行前需要可用的云账号、容器镜像仓库、kubemark 二进制和目标 provider 配置。涉及资源模板、hollow node 参数或 e2e 参数的改动，应在小规模 Kubemark 集群上先验证创建、测试和清理流程。
