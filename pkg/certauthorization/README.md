# certauthorization 目录说明

`pkg/certauthorization` 提供证书签名器名称相关的授权辅助逻辑，用于判断用户是否有权限对 `certificates.k8s.io` API 中的 `signers` 资源执行指定操作。

## 关键子目录/源码入口

- `certauthorization.go`：实现 `IsAuthorizedForSignerName`，先检查完整 signerName，再检查域名前缀通配权限。

## 与 cmd/staging/test 的关系

- kube-apiserver 证书相关 REST 或控制逻辑可调用本包完成 signer 授权判断。
- 授权接口来自 `staging/src/k8s.io/apiserver/pkg/authorization/authorizer`，本目录只封装 Kubernetes 证书 API 的资源属性构造规则。
- 相关行为通常由调用方单元测试或证书 API 集成测试覆盖。

## 开发/测试注意事项

- 修改授权属性时，要保持 API group、resource、verb 和 name 与 RBAC 规则一致。
- 通配授权只应使用 signerName 的域名部分，例如 `kubernetes.io/*`。
- 证书授权变更容易影响安全边界，应补充允许、拒绝、错误和通配场景测试。
