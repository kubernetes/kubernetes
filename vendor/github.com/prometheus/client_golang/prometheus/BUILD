load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "collector.go",
        "counter.go",
        "desc.go",
        "doc.go",
        "expvar_collector.go",
        "fnv.go",
        "gauge.go",
        "go_collector.go",
        "histogram.go",
        "http.go",
        "metric.go",
        "observer.go",
        "process_collector.go",
        "registry.go",
        "summary.go",
        "timer.go",
        "untyped.go",
        "value.go",
        "vec.go",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/beorn7/perks/quantile:go_default_library",
        "//vendor/github.com/golang/protobuf/proto:go_default_library",
        "//vendor/github.com/prometheus/client_model/go:go_default_library",
        "//vendor/github.com/prometheus/common/expfmt:go_default_library",
        "//vendor/github.com/prometheus/common/model:go_default_library",
        "//vendor/github.com/prometheus/procfs:go_default_library",
    ],
)

filegroup(
    name = "package-srcs",
    srcs = glob(["**"]),
    tags = ["automanaged"],
    visibility = ["//visibility:private"],
)

filegroup(
    name = "all-srcs",
    srcs = [
        ":package-srcs",
        "//vendor/github.com/prometheus/client_golang/prometheus/promhttp:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
