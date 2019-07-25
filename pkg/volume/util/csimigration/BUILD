load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["plugins.go"],
    importpath = "k8s.io/kubernetes/pkg/volume/util/csimigration",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/features:go_default_library",
        "//pkg/volume:go_default_library",
        "//pkg/volume/awsebs:go_default_library",
        "//pkg/volume/azure_dd:go_default_library",
        "//pkg/volume/azure_file:go_default_library",
        "//pkg/volume/cinder:go_default_library",
        "//pkg/volume/gcepd:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/util/feature:go_default_library",
        "//staging/src/k8s.io/component-base/featuregate:go_default_library",
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
