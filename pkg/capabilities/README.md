# capabilities 目录说明

`pkg/capabilities` 管理 Kubernetes 进程级能力开关，用于记录和查询当前进程支持的系统能力集合。

## 关键子目录/源码入口

- `capabilities.go`：定义 capability 集合、默认值和全局读写入口。
- `capabilities_test.go`：覆盖 capability 设置和读取行为。

## 与 cmd/staging/test 的关系

- `cmd` 下组件可通过本包读取进程能力信息，以决定是否启用依赖宿主能力的行为。
- 本目录不属于 staging 公共 API，通用 runtime 或组件基础库不应反向依赖它。
- 测试主要是包内单元测试。

## 开发/测试注意事项

- 这里维护的是进程级状态，修改时要避免引入测试间污染。
- 新增能力标识时，应确认调用方的默认行为和降级路径。
- 建议运行本包测试，并关注并发访问或全局状态重置场景。
