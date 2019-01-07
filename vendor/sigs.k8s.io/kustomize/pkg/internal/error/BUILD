load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "configmaperror.go",
        "kustomizationerror.go",
        "patcherror.go",
        "resourceerror.go",
        "secreterror.go",
        "yamlformaterror.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/sigs.k8s.io/kustomize/pkg/internal/error",
    importpath = "sigs.k8s.io/kustomize/pkg/internal/error",
    visibility = ["//vendor/sigs.k8s.io/kustomize/pkg:__subpackages__"],
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
