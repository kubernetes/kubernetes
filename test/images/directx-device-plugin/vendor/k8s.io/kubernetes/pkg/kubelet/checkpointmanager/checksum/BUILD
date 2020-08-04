load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["checksum.go"],
    importpath = "k8s.io/kubernetes/pkg/kubelet/checkpointmanager/checksum",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/kubelet/checkpointmanager/errors:go_default_library",
        "//pkg/util/hash:go_default_library",
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
