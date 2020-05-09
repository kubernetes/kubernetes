load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["logs.go"],
    importpath = "k8s.io/kubernetes/pkg/kubelet/legacy",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/kubelet/container:go_default_library",
        "//pkg/kubelet/kuberuntime:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
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
