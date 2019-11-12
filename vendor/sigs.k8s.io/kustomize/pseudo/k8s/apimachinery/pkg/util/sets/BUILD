load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "byte.go",
        "doc.go",
        "empty.go",
        "int.go",
        "int32.go",
        "int64.go",
        "string.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/sigs.k8s.io/kustomize/pseudo/k8s/apimachinery/pkg/util/sets",
    importpath = "sigs.k8s.io/kustomize/pseudo/k8s/apimachinery/pkg/util/sets",
    visibility = ["//visibility:public"],
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
