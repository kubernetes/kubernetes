# windows 目录说明

`pkg/windows` 保存 Kubernetes Windows 平台专用的内部辅助代码，目前主要包含 Windows service 集成，用于支持组件以 Windows 服务方式运行。

关键入口和子目录：
- `service/service.go` 封装 Windows service 的安装、运行和控制处理逻辑。
- `service/OWNERS` 标识该子目录维护者。

与 `cmd`、`staging`、`test` 的关系：
- Windows 平台上的组件命令入口可复用这里的 service 辅助能力来接入系统服务管理器。
- 本目录偏向主仓库内部平台适配，公共跨仓库 API 应优先放在合适的 staging 模块。
- Windows 专项测试通常与源码同目录或在节点/e2e 测试中覆盖，非 Windows 平台需要保持构建隔离。

开发和测试注意事项：
- 修改时要关注 Windows 构建标签、服务控制事件、日志和进程生命周期。
- 不应向非 Windows 构建路径引入 Windows-only 依赖。
- 需要在 Windows 环境验证服务安装、启动、停止和异常退出行为。
