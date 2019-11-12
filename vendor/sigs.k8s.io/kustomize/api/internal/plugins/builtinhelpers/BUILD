load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "builtinplugintype_string.go",
        "builtins.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/sigs.k8s.io/kustomize/api/internal/plugins/builtinhelpers",
    importpath = "sigs.k8s.io/kustomize/api/internal/plugins/builtinhelpers",
    visibility = ["//vendor/sigs.k8s.io/kustomize/api:__subpackages__"],
    deps = [
        "//vendor/sigs.k8s.io/kustomize/api/builtins:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/api/resmap:go_default_library",
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
