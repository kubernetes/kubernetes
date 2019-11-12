load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["validation.go"],
    importmap = "k8s.io/kubernetes/vendor/sigs.k8s.io/kustomize/pseudo/k8s/apimachinery/pkg/util/validation",
    importpath = "sigs.k8s.io/kustomize/pseudo/k8s/apimachinery/pkg/util/validation",
    visibility = ["//visibility:public"],
    deps = ["//vendor/sigs.k8s.io/kustomize/pseudo/k8s/apimachinery/pkg/util/validation/field:go_default_library"],
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
        "//vendor/sigs.k8s.io/kustomize/pseudo/k8s/apimachinery/pkg/util/validation/field:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
