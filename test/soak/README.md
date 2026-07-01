# test/soak 目录说明

`test/soak` 存放长时间运行的稳定性测试程序，用于持续施加真实 Kubernetes API 和数据面流量，观察集群在较长时间内是否保持健康。当前主要示例是 `serve_hostnames`，它在每个节点上创建 Pod，通过 Service 持续请求并验证所有 Pod 都能响应。

## 关键子目录/源码入口

- `serve_hostnames/`：长期运行的主机名服务测试。
- `serve_hostnames/serve_hostnames.go`：测试程序入口，读取 kubeconfig，枚举节点，创建命名空间、Service 和 Pod，并按参数持续发起请求。
- `serve_hostnames/README.md`：该具体 soak 测试的运行方式、参数和示例输出。
- `serve_hostnames/Makefile`：构建该测试程序的入口。

## 与 Kubernetes 其他模块的关系

Soak 测试通过 `client-go` 操作真实集群资源，并复用 `test/e2e/framework/service`、`test/utils/image` 和核心 API 类型。它覆盖 API Server、调度、kubelet、Service proxy、Pod 网络和清理路径的组合行为，可作为 e2e 测试之外的长稳验证补充。

## 开发/测试注意事项

这类测试会在真实集群中创建和删除资源，运行前确认 kubeconfig 指向预期集群，并理解 `--up_to=-1` 等参数可能导致无限运行。新增 soak 测试应限制并发和资源规模，确保失败时能清理命名空间、Pod 和 Service。测试目标是稳定性和健康检查，不应把瞬时 QPS 当作正式性能基准。
