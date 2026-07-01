# genfeaturegates 目录说明

`cmd/genfeaturegates/` 用于生成 Kubernetes feature gate 清单。它加载 apiserver、apiextensions-apiserver 和 Kubernetes 主仓库中注册的 feature gate，并按名称、阶段或版本输出 Markdown 或 JSON。

## 关键文件

- `genfeaturegates.go`：程序入口和生成逻辑，支持排序、反向排序、阶段过滤、输出格式和输出文件参数。

## 与其他模块的关系

该工具依赖 `k8s.io/component-base/featuregate` 以及各模块通过空白导入注册的 feature gate。它把分散在 `pkg/features`、`k8s.io/apiserver/pkg/features`、`k8s.io/apiextensions-apiserver/pkg/features` 等位置的特性开关汇总为文档数据。

## 开发与测试注意事项

新增或调整 feature gate 时，应同时确认阶段、默认值、锁定状态、依赖关系和版本范围是否准确。生成结果可用于文档站点或验证流程；如果输出文件在子目录中，工具会自动创建目录，但仍应避免把临时路径作为仓库产物提交。
