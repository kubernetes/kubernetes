package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = ["install.go"],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/apiserver/pkg/apis/apiserver/install",
    importpath = "k8s.io/apiserver/pkg/apis/apiserver/install",
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/runtime:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/apis/apiserver:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/apis/apiserver/v1:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/apis/apiserver/v1alpha1:go_default_library",
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
)
