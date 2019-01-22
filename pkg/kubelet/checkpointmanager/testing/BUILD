package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = ["util.go"],
    importpath = "k8s.io/kubernetes/pkg/kubelet/checkpointmanager/testing",
)

filegroup(
    name = "package-srcs",
    srcs = glob(["**"]),
    tags = ["automanaged"],
    visibility = ["//visibility:private"],
)

filegroup(
    name = "all-srcs",
    srcs = [
        ":package-srcs",
        "//pkg/kubelet/checkpointmanager/testing/example_checkpoint_formats/v1:all-srcs",
    ],
    tags = ["automanaged"],
)
