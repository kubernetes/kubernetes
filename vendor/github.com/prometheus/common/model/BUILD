load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "alert.go",
        "fingerprinting.go",
        "fnv.go",
        "labels.go",
        "labelset.go",
        "metric.go",
        "model.go",
        "signature.go",
        "silence.go",
        "time.go",
        "value.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/prometheus/common/model",
    importpath = "github.com/prometheus/common/model",
    visibility = ["//build/visible_to:vendor_githubcom_prometheus_CONSUMERS"],
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
