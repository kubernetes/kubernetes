# kube-scheduler 目录说明

`cmd/kube-scheduler/` 是 Kubernetes Scheduler 二进制入口目录。调度器监听未绑定节点的 Pod，基于调度框架、插件和策略选择合适节点，并把调度结果写回 API Server。

## 关键文件和子目录

- `scheduler.go`：`main` 入口，注册日志和指标插件，创建并运行 scheduler 命令。
- `app/`：命令构造、配置加载、调度框架初始化、插件注册和调度器运行入口。
- `OWNERS`：维护该组件入口的代码所有者信息。

## 与其他模块的关系

该组件与 kube-apiserver、client-go informer、`pkg/scheduler`、调度插件、feature gate 和 component-base 配置体系协作。Pod、Node、Volume、ResourceClaim 等 API 的变化可能影响调度行为。

## 开发与测试注意事项

改动调度插件、默认配置或调度队列行为时，要关注性能、可预测性和升级兼容性。通常需要覆盖单元测试、集成测试或调度性能测试。命令 flag 和配置字段变化还会影响文档生成与配置 API。
