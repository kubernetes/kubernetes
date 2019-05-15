load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["checkpoint.go"],
    importpath = "k8s.io/kubernetes/cmd/kubelet/pkg/cm/devicemanager/checkpoint",
    visibility = ["//visibility:public"],
    deps = [
        "//cmd/kubelet/pkg/checkpointmanager:go_default_library",
        "//cmd/kubelet/pkg/checkpointmanager/checksum:go_default_library",
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
