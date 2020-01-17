load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "build_info.go",
        "build_info_pre_1.12.go",
        "collector.go",
        "counter.go",
        "desc.go",
        "doc.go",
        "expvar_collector.go",
        "fnv.go",
        "gauge.go",
        "go_collector.go",
        "histogram.go",
        "labels.go",
        "metric.go",
        "observer.go",
        "process_collector.go",
        "process_collector_other.go",
        "process_collector_windows.go",
        "registry.go",
        "summary.go",
        "timer.go",
        "untyped.go",
        "value.go",
        "vec.go",
        "wrap.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/prometheus/client_golang/prometheus",
    importpath = "github.com/prometheus/client_golang/prometheus",
    visibility = ["//staging/src/k8s.io/component-base/metrics:prometheus_import_allow_list"],
    deps = [
        "//vendor/github.com/beorn7/perks/quantile:go_default_library",
        "//vendor/github.com/golang/protobuf/proto:go_default_library",
        "//vendor/github.com/prometheus/client_golang/prometheus/internal:go_default_library",
        "//vendor/github.com/prometheus/client_model/go:go_default_library",
        "//vendor/github.com/prometheus/common/expfmt:go_default_library",
        "//vendor/github.com/prometheus/common/model:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "//vendor/github.com/prometheus/procfs:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "//vendor/github.com/prometheus/procfs:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "//vendor/github.com/prometheus/procfs:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "//vendor/github.com/prometheus/procfs:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:ios": [
            "//vendor/github.com/prometheus/procfs:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "//vendor/github.com/prometheus/procfs:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "//vendor/github.com/prometheus/procfs:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "//vendor/github.com/prometheus/procfs:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "//vendor/github.com/prometheus/procfs:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "//vendor/github.com/prometheus/procfs:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "//vendor/github.com/prometheus/procfs:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "//vendor/golang.org/x/sys/windows:go_default_library",
        ],
        "//conditions:default": [],
    }),
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
        "//vendor/github.com/prometheus/client_golang/prometheus/internal:all-srcs",
        "//vendor/github.com/prometheus/client_golang/prometheus/promhttp:all-srcs",
        "//vendor/github.com/prometheus/client_golang/prometheus/testutil:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
