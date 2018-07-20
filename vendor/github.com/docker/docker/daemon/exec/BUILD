load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["exec.go"],
    importmap = "k8s.io/kubernetes/vendor/github.com/docker/docker/daemon/exec",
    importpath = "github.com/docker/docker/daemon/exec",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/containerd/containerd/cio:go_default_library",
        "//vendor/github.com/docker/docker/container/stream:go_default_library",
        "//vendor/github.com/docker/docker/libcontainerd:go_default_library",
        "//vendor/github.com/docker/docker/pkg/stringid:go_default_library",
        "//vendor/github.com/sirupsen/logrus:go_default_library",
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
