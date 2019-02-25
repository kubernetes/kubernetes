load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["install.go"],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/metrics/pkg/apis/custom_metrics/install",
    importpath = "k8s.io/metrics/pkg/apis/custom_metrics/install",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/runtime:go_default_library",
        "//staging/src/k8s.io/metrics/pkg/apis/custom_metrics:go_default_library",
        "//staging/src/k8s.io/metrics/pkg/apis/custom_metrics/v1beta1:go_default_library",
        "//staging/src/k8s.io/metrics/pkg/apis/custom_metrics/v1beta2:go_default_library",
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
