package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = ["types.go"],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/kubelet/pkg/apis/stats/v1alpha1",
    importpath = "k8s.io/kubelet/pkg/apis/stats/v1alpha1",
    deps = ["//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library"],
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
