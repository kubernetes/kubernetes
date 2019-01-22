load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "fds_linux.go",
        "fds_other.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/coreos/etcd/pkg/runtime",
    importpath = "github.com/coreos/etcd/pkg/runtime",
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
