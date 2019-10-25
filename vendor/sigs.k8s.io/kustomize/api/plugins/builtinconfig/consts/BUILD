load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "commonannotations.go",
        "commonlabels.go",
        "defaultconfig.go",
        "doc.go",
        "images.go",
        "nameprefix.go",
        "namereference.go",
        "namespace.go",
        "replicas.go",
        "varreference.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/sigs.k8s.io/kustomize/api/plugins/builtinconfig/consts",
    importpath = "sigs.k8s.io/kustomize/api/plugins/builtinconfig/consts",
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
