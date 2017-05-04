package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "args.go",
        "delegate.go",
        "exec.go",
        "find.go",
        "os_unix.go",
        "raw_exec.go",
    ],
    tags = ["automanaged"],
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
)
