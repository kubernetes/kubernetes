# scheduler 目录说明

`pkg/scheduler` 是 kube-scheduler 的核心实现目录，负责调度队列、调度循环、插件框架、缓存、抢占和默认调度插件。

关键入口和子目录：
- `scheduler.go`、`schedule_one.go`、`eventhandlers.go` 是调度器主流程、单 Pod 调度周期和对象事件处理入口。
- `framework/` 定义调度插件扩展点、运行时、CycleState、抢占和 API 调用封装。
- `framework/plugins/` 保存默认调度插件，例如资源匹配、亲和性、拓扑分布、卷绑定和抢占。
- `backend/` 提供调度队列、缓存、heap 和 API cache 等内部数据结构。
- `apis/config/` 定义 kube-scheduler 配置 API、默认值、转换和校验。

与 `cmd`、`staging`、`test` 的关系：
- `cmd/kube-scheduler` 负责命令行、配置加载和依赖构建，随后启动本目录的调度器。
- 调度器依赖 staging 中的 API、client-go、component-base 和 framework 相关通用能力。
- 单元测试覆盖队列、缓存、插件和调度周期；调度行为还由 `test/e2e`、集成测试和性能测试覆盖。

开发和测试注意事项：
- 修改调度循环、队列或缓存时要关注并发、事件重入、PodGroup、抢占和性能退化。
- 新增插件或配置字段时要同步默认插件集合、配置 API、校验和测试数据。
- 调度结果是用户可见行为，算法调整需要明确兼容性、可解释性和可测试性。
