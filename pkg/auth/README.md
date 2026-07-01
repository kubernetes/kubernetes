# auth 目录说明

`pkg/auth` 放置 Kubernetes 仓库内的认证/授权辅助实现，目前主要包含 ABAC 授权器和节点身份识别工具。

## 关键子目录/源码入口

- `authorizer/abac/`：实现基于属性的访问控制策略加载和授权判断，并包含示例策略文件。
- `nodeidentifier/`：识别请求用户是否对应 Kubernetes Node，并提取节点名称。

## 与 cmd/staging/test 的关系

- `cmd/kube-apiserver` 的认证授权配置可间接使用这些实现。
- 通用认证、授权接口来自 `staging/src/k8s.io/apiserver`，本目录提供 Kubernetes 特定实现或适配。
- 测试主要位于各子包内，授权链路也可能被 kube-apiserver 集成测试覆盖。

## 开发/测试注意事项

- 授权相关代码要保持拒绝优先和错误可观测，避免在解析失败时意外放行。
- 修改 ABAC 策略解析时，应覆盖注释、空行、版本化策略和错误行号。
- 修改节点身份逻辑时，需要同步考虑 Node authorizer、Node admission 和证书用户名约定。
