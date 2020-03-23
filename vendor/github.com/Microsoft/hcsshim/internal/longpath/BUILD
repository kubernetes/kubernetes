load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["longpath.go"],
    importmap = "k8s.io/kubernetes/vendor/github.com/Microsoft/hcsshim/internal/longpath",
    importpath = "github.com/Microsoft/hcsshim/internal/longpath",
    visibility = ["//vendor/github.com/Microsoft/hcsshim:__subpackages__"],
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
