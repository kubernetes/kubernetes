load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["services.go"],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/cri-api/pkg/apis",
    importpath = "k8s.io/cri-api/pkg/apis",
    visibility = ["//visibility:public"],
    deps = ["//staging/src/k8s.io/cri-api/pkg/apis/runtime/v1alpha2:go_default_library"],
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
        "//staging/src/k8s.io/cri-api/pkg/apis/runtime/v1alpha2:all-srcs",
        "//staging/src/k8s.io/cri-api/pkg/apis/testing:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
