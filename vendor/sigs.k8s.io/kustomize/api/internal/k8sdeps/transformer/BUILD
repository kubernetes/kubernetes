load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["factory.go"],
    importmap = "k8s.io/kubernetes/vendor/sigs.k8s.io/kustomize/api/internal/k8sdeps/transformer",
    importpath = "sigs.k8s.io/kustomize/api/internal/k8sdeps/transformer",
    visibility = ["//vendor/sigs.k8s.io/kustomize/api:__subpackages__"],
    deps = [
        "//vendor/sigs.k8s.io/kustomize/api/internal/k8sdeps/transformer/patch:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/api/resmap:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/api/resource:go_default_library",
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
        "//vendor/sigs.k8s.io/kustomize/api/internal/k8sdeps/transformer/patch:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
