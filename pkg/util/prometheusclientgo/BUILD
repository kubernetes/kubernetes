load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["adapters.go"],
    importpath = "k8s.io/kubernetes/pkg/util/prometheusclientgo",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/client/metrics/prometheus:go_default_library",
        "//pkg/util/prometheusclientgo/leaderelection:go_default_library",
        "//pkg/util/workqueue/prometheus:go_default_library",
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
        "//pkg/util/prometheusclientgo/leaderelection:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
