# volume 目录说明

`pkg/volume` 是 kubelet 内部卷插件和卷操作的核心目录，负责内置 volume 类型、CSI、挂载/卸载、扩容、指标、校验和测试辅助。

关键入口和子目录：
- `volume.go` 定义 volume 插件、mounter、unmounter、attacher、detacher 等核心接口。
- `plugins.go` 负责注册和初始化内置卷插件。
- `csi/`、`configmap/`、`secret/`、`projected/`、`emptydir/`、`hostpath/`、`local/`、`nfs/`、`iscsi/`、`fc/`、`flexvolume/` 等目录实现具体卷类型。
- `util/` 提供子路径、文件系统配额、host util、operation executor、resize、atomic writer 等共享工具。
- `validation/` 和 `testing/` 分别提供 PV 相关校验和测试辅助。

与 `cmd`、`staging`、`test` 的关系：
- `cmd/kubelet` 启动后通过 `pkg/kubelet/volumemanager` 调用本目录的插件和操作能力。
- CSI、存储 API、client 和 mount 工具大量来自 staging 模块；外部存储接口应保持与 staging API 一致。
- 单元测试与插件源码同目录；真实挂载、CSI 和存储行为还依赖 e2e、e2e node 及存储专项测试。

开发和测试注意事项：
- 卷代码涉及宿主机文件系统和设备操作，修改时必须关注清理路径、幂等性、权限、SELinux 和 Windows/Linux 差异。
- 新增或调整插件时要覆盖 attach/detach、mount/unmount、reconstruct、resize 和错误恢复场景。
- 避免在插件中绕过 kubelet volume manager 的状态机和 operation executor。
