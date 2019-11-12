load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "converter.go",
        "deep_equal.go",
        "doc.go",
        "helper.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/sigs.k8s.io/kustomize/pseudo/k8s/apimachinery/pkg/conversion",
    importpath = "sigs.k8s.io/kustomize/pseudo/k8s/apimachinery/pkg/conversion",
    visibility = ["//visibility:public"],
    deps = ["//vendor/sigs.k8s.io/kustomize/pseudo/k8s/apimachinery/third_party/forked/golang/reflect:go_default_library"],
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
        "//vendor/sigs.k8s.io/kustomize/pseudo/k8s/apimachinery/pkg/conversion/queryparams:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
