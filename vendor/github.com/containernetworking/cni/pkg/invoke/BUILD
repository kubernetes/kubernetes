load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "args.go",
        "delegate.go",
        "exec.go",
        "find.go",
        "os_unix.go",
        "os_windows.go",
        "raw_exec.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/containernetworking/cni/pkg/invoke",
    importpath = "github.com/containernetworking/cni/pkg/invoke",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/containernetworking/cni/pkg/types:go_default_library",
        "//vendor/github.com/containernetworking/cni/pkg/version:go_default_library",
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
