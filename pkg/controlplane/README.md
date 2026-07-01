# controlplane 目录说明

`pkg/controlplane` 包含组装和运行 Kubernetes 控制面 API server 的核心代码，负责 API 安装、REST storage 装配、控制面控制器和端点协调等逻辑。

## 关键子目录/源码入口

- `instance.go`：构造 control plane 实例，安装 API 组和 REST storage。
- `apiserver/`：封装 Kubernetes API server 的配置、完成、启动和聚合相关逻辑。
- `apiserver/options/`：control plane API server 选项与校验。
- `controller/`：控制面内部控制器，例如 kubernetes Service、默认 ServiceCIDR、leader election、系统命名空间和 apiserver lease GC。
- `reconcilers/`：维护 apiserver identity、endpoint 和 lease 等协调逻辑。

## 与 cmd/staging/test 的关系

- `cmd/kube-apiserver` 主要作为入口，实际控制面装配逻辑位于本目录和 `pkg/kubeapiserver`。
- 通用 apiserver 框架来自 `staging/src/k8s.io/apiserver`，本目录负责 Kubernetes API 组、registry 和控制面控制器的具体接线。
- 包内测试、sample apiserver 测试和 kube-apiserver 集成测试共同覆盖启动和配置路径。

## 开发/测试注意事项

- 修改 API 安装或 REST storage 时，要同步考虑 discovery、存储版本、feature gate 和降级兼容。
- 控制面控制器通常依赖 informer、lease 和 Service 语义，需保证启动顺序和 leader election 行为。
- 建议运行相关 `pkg/controlplane` 测试，并根据影响范围运行 kube-apiserver 集成测试。
