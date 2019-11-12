load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "fields.go",
        "requirements.go",
        "selector.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/sigs.k8s.io/kustomize/pseudo/k8s/apimachinery/pkg/fields",
    importpath = "sigs.k8s.io/kustomize/pseudo/k8s/apimachinery/pkg/fields",
    visibility = ["//visibility:public"],
    deps = ["//vendor/sigs.k8s.io/kustomize/pseudo/k8s/apimachinery/pkg/selection:go_default_library"],
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
