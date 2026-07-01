# controller 目录说明

`pkg/controller` 包含 kube-controller-manager 运行的大量 Kubernetes 控制器实现，以及控制器共享的工具函数、配置类型、指标和测试。

## 关键子目录/源码入口

- `deployment/`、`replicaset/`、`statefulset/`、`daemon/`、`job/`、`cronjob/`：工作负载控制器。
- `node*`、`namespace/`、`resourcequota/`、`serviceaccount/`、`garbagecollector/`：集群资源和生命周期控制器。
- `volume/`、`persistentvolume/`、`attachdetach/`、`expand/`：存储相关控制器。
- `controller_utils.go` 和各控制器的 `config/`、`metrics/` 子目录：共享工具、配置 API 和指标定义。

## 与 cmd/staging/test 的关系

- `cmd/kube-controller-manager` 负责装配和启动这些控制器。
- 通用 controller-manager 框架、leader election、workqueue、client-go informer 来自 staging 模块，本目录实现 Kubernetes 资源的具体 reconcile 逻辑。
- 单元测试分布在各控制器目录，端到端和集成测试会覆盖跨组件行为。

## 开发/测试注意事项

- 控制器变更要关注幂等性、重试、事件顺序、缓存延迟和并发 worker 行为。
- 修改 API 写入路径时，需要确认 owner reference、finalizer、status/update 子资源和冲突重试。
- 建议运行对应控制器包测试；影响共享工具或核心控制器时，补充集成测试。
