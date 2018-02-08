load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["interface.go"],
    importpath = "k8s.io/apiextensions-apiserver/examples/client-go/pkg/client/informers/externalversions/cr",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/k8s.io/apiextensions-apiserver/examples/client-go/pkg/client/informers/externalversions/cr/v1:go_default_library",
        "//vendor/k8s.io/apiextensions-apiserver/examples/client-go/pkg/client/informers/externalversions/internalinterfaces:go_default_library",
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
        "//staging/src/k8s.io/apiextensions-apiserver/examples/client-go/pkg/client/informers/externalversions/cr/v1:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
