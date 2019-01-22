package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = ["services.go"],
    importpath = "k8s.io/kubernetes/pkg/kubelet/apis/cri",
    deps = ["//pkg/kubelet/apis/cri/runtime/v1alpha2:go_default_library"],
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
        "//pkg/kubelet/apis/cri/runtime/v1alpha2:all-srcs",
        "//pkg/kubelet/apis/cri/testing:all-srcs",
    ],
    tags = ["automanaged"],
)
