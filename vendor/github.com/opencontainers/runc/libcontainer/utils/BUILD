load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "cmsg.go",
        "utils.go",
        "utils_unix.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/opencontainers/runc/libcontainer/utils",
    importpath = "github.com/opencontainers/runc/libcontainer/utils",
    visibility = ["//visibility:public"],
    deps = ["//vendor/golang.org/x/sys/unix:go_default_library"],
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
