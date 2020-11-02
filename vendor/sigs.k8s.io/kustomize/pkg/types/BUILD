load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "genargs.go",
        "generationbehavior.go",
        "kustomization.go",
        "var.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/sigs.k8s.io/kustomize/pkg/types",
    importpath = "sigs.k8s.io/kustomize/pkg/types",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/sigs.k8s.io/kustomize/pkg/gvk:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pkg/image:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pkg/patch:go_default_library",
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
