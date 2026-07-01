# kubelet 目录说明

`cmd/kubelet/` 是 kubelet 节点代理二进制入口目录。kubelet 运行在每个节点上，负责接收 Pod 配置、驱动容器运行时、管理卷和网络相关状态，并向 API Server 汇报节点与 Pod 状态。

## 关键文件和子目录

- `kubelet.go`：`main` 入口，注册日志和指标插件，创建并运行 kubelet 命令。
- `app/`：命令选项、配置加载、认证授权、平台初始化、插件初始化和 kubelet server 启动逻辑。
- `OWNERS`：维护该组件入口的代码所有者信息。

## 与其他模块的关系

kubelet 连接 kube-apiserver、container runtime、CNI、CSI、device plugin、credential provider、节点状态管理和 `pkg/kubelet` 核心逻辑。它是控制面期望状态落地到节点运行时的关键组件。

## 开发与测试注意事项

改动 kubelet 启动、配置或认证逻辑时要关注节点升级、配置文件兼容性、平台差异和与容器运行时的交互。相关测试可能需要覆盖单元测试、节点 e2e、Windows/Linux 差异以及 feature gate 组合。
