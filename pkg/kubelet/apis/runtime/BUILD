load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["services.go"],
    importpath = "k8s.io/kubernetes/pkg/kubelet/apis/runtime",
    visibility = ["//visibility:public"],
    deps = ["//pkg/kubelet/apis/runtime/v1alpha2:go_default_library"],
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
        "//pkg/kubelet/apis/runtime/testing:all-srcs",
        "//pkg/kubelet/apis/runtime/v1alpha2:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
