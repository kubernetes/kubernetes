load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "errors.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/sigs.k8s.io/kustomize/pseudo/k8s/apimachinery/pkg/util/errors",
    importpath = "sigs.k8s.io/kustomize/pseudo/k8s/apimachinery/pkg/util/errors",
    visibility = ["//visibility:public"],
    deps = ["//vendor/sigs.k8s.io/kustomize/pseudo/k8s/apimachinery/pkg/util/sets:go_default_library"],
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
