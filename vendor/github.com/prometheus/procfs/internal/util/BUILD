load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "parse.go",
        "readfile.go",
        "sysreadfile.go",
        "sysreadfile_compat.go",
        "valueparser.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/prometheus/procfs/internal/util",
    importpath = "github.com/prometheus/procfs/internal/util",
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
