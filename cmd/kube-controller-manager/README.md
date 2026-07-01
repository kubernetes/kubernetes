# kube-controller-manager 目录说明

`cmd/kube-controller-manager/` 是 Kubernetes Controller Manager 二进制入口目录。它运行多种内置控制器，通过 watch API 对象并持续调谐集群状态，使实际状态接近期望状态。

## 关键文件和子目录

- `controller-manager.go`：`main` 入口，注册日志和指标插件，创建并运行 controller manager 命令。
- `app/`：命令选项、控制器注册、控制器启动、健康检查和各 API 领域控制器接线逻辑。
- `OWNERS`：维护该组件入口的代码所有者信息。

## 与其他模块的关系

该组件依赖 kube-apiserver 提供的 API，通过 client-go informer 和 workqueue 驱动控制循环。`app/` 中的控制器接线会连接 `pkg/controller`、各 API group、事件记录、leader election 和 component-base 配置体系。

## 开发与测试注意事项

新增或调整控制器时，要确认默认启用状态、feature gate、RBAC、leader election 和事件行为。相关变更通常需要单元测试，涉及真实 API 交互时还应考虑集成测试。命令行 flag 变化会影响生成文档和组件配置兼容性。
