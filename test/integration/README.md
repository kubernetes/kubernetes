# test/integration 目录说明

`test/integration` 包含 Kubernetes 核心组件的集成测试公共包和测试资源，用于在较真实的组件组合下验证 API Server、存储、认证授权、控制器和客户端行为。许多具体集成测试分布在 `test/integration/*` 子包或 staging 模块的 `test/integration` 下。

## 关键子目录/源码入口

- `doc.go`：说明 `integration` 包用途，提示部分测试需要本机可用的 etcd 或 Docker。
- `utils.go`：提供常用集成测试辅助函数，例如删除 Pod、等待 Pod 消失、获取 etcd client 等。
- `.import-restrictions`：约束该测试区域的导入边界。
- `benchmark-controller.json`：集成测试/基准场景使用的控制器配置资源。

## 与 Kubernetes 其他模块的关系

该目录与 `cmd/kube-apiserver/app/testing`、`k8s.io/apiserver`、`k8s.io/client-go`、`k8s.io/apimachinery` 和 etcd 存储层紧密相关。集成测试通常启动或连接测试 API Server，并通过真实 clientset 操作 Kubernetes API，以验证单元测试覆盖不到的跨模块交互。

## 开发/测试注意事项

新增公共 helper 时要保持通用、稳定，并避免把特定测试场景的假设放入共享包。运行集成测试前确认 etcd、Docker 或测试二进制依赖是否满足；涉及 API Server 存储、认证授权、准入或控制器行为的改动，应运行对应子包的集成测试。测试要正确清理命名空间、对象和后台 goroutine，防止影响同进程后续用例。
