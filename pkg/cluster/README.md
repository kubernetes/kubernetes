# cluster 目录说明

`pkg/cluster` 放置集群级别的常量和基础设施约定，目前主要维护 Kubernetes 各组件使用的端口定义。

## 关键子目录/源码入口

- `ports/`：定义 kube-apiserver、kubelet、controller-manager、scheduler 等组件或服务使用的默认端口。

## 与 cmd/staging/test 的关系

- `cmd` 下组件和测试环境可引用这些端口常量，避免散落魔法数字。
- 本目录是 Kubernetes 仓库内部约定，不属于 staging 中的稳定外部 API。
- 端口常量通常通过引用方测试或配置测试间接覆盖。

## 开发/测试注意事项

- 修改默认端口可能影响部署脚本、测试集群和文档，需确认兼容性。
- 新增端口常量时应说明用途，并避免与已有组件默认端口冲突。
- 建议运行引用该常量的组件配置测试。
