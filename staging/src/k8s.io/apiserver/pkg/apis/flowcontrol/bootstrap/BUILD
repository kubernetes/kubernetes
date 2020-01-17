load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["default.go"],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/apiserver/pkg/apis/flowcontrol/bootstrap",
    importpath = "k8s.io/apiserver/pkg/apis/flowcontrol/bootstrap",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/api/coordination/v1:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/api/flowcontrol/v1alpha1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/authentication/serviceaccount:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/authentication/user:go_default_library",
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
