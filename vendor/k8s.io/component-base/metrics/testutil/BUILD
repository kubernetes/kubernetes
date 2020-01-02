load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "metrics.go",
        "testutil.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/component-base/metrics/testutil",
    importpath = "k8s.io/component-base/metrics/testutil",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/component-base/metrics:go_default_library",
        "//vendor/github.com/prometheus/client_golang/prometheus/testutil:go_default_library",
        "//vendor/github.com/prometheus/common/expfmt:go_default_library",
        "//vendor/github.com/prometheus/common/model:go_default_library",
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
    srcs = [":package-srcs"],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
