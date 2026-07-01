# kube-proxy 目录说明

`cmd/kube-proxy/` 是 kube-proxy 二进制入口目录。kube-proxy 运行在节点上，监听 Service、EndpointSlice 等对象，并配置节点本地转发规则以实现 Kubernetes Service 访问。

## 关键文件和子目录

- `proxy.go`：`main` 入口，注册日志和指标插件，创建并运行 kube-proxy 命令。
- `app/`：命令选项、配置加载、平台初始化和代理模式启动逻辑，包含 Linux、Windows 和其他平台的差异实现。
- `OWNERS`：维护该组件入口的代码所有者信息。

## 与其他模块的关系

kube-proxy 依赖 kube-apiserver 获取服务发现数据，并与 `pkg/proxy`、client-go informer、node 网络栈和平台相关代理实现协作。它的行为直接影响 Service、NodePort、ClusterIP、external traffic policy 等网络功能。

## 开发与测试注意事项

网络路径变更需要同时考虑 Linux 和 Windows 差异、iptables/ipvs/nftables 等模式、双栈和 EndpointSlice 行为。修改配置字段或 flag 时要注意组件配置兼容性，并运行相关单元测试及平台特定测试。
