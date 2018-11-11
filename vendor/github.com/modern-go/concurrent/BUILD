load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "executor.go",
        "go_above_19.go",
        "go_below_19.go",
        "log.go",
        "unbounded_executor.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/modern-go/concurrent",
    importpath = "github.com/modern-go/concurrent",
    visibility = ["//visibility:public"],
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
