load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "maptransformer.go",
        "multitransformer.go",
        "mutatefield.go",
        "nooptransformer.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/sigs.k8s.io/kustomize/api/transform",
    importpath = "sigs.k8s.io/kustomize/api/transform",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/sigs.k8s.io/kustomize/api/resmap:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/api/types:go_default_library",
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
