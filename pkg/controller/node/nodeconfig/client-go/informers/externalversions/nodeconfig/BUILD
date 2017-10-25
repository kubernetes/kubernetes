load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["interface.go"],
    importpath = "k8s.io/kubernetes/pkg/controller/node/nodeconfig/client-go/informers/externalversions/nodeconfig",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/controller/node/nodeconfig/client-go/informers/externalversions/internalinterfaces:go_default_library",
        "//pkg/controller/node/nodeconfig/client-go/informers/externalversions/nodeconfig/v1alpha1:go_default_library",
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
        "//pkg/controller/node/nodeconfig/client-go/informers/externalversions/nodeconfig/v1alpha1:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
