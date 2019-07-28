load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "api_linter.go",
        "config.go",
        "extension.go",
        "openapi.go",
        "union.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/kube-openapi/pkg/generators",
    importpath = "k8s.io/kube-openapi/pkg/generators",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/k8s.io/gengo/args:go_default_library",
        "//vendor/k8s.io/gengo/examples/set-gen/sets:go_default_library",
        "//vendor/k8s.io/gengo/generator:go_default_library",
        "//vendor/k8s.io/gengo/namer:go_default_library",
        "//vendor/k8s.io/gengo/types:go_default_library",
        "//vendor/k8s.io/klog:go_default_library",
        "//vendor/k8s.io/kube-openapi/cmd/openapi-gen/args:go_default_library",
        "//vendor/k8s.io/kube-openapi/pkg/common:go_default_library",
        "//vendor/k8s.io/kube-openapi/pkg/generators/rules:go_default_library",
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
    srcs = [
        ":package-srcs",
        "//vendor/k8s.io/kube-openapi/pkg/generators/rules:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
