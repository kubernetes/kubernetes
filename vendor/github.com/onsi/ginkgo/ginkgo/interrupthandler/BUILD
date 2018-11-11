load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "interrupt_handler.go",
        "sigquit_swallower_unix.go",
        "sigquit_swallower_windows.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/onsi/ginkgo/ginkgo/interrupthandler",
    importpath = "github.com/onsi/ginkgo/ginkgo/interrupthandler",
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
