package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = ["interface.go"],
    importpath = "k8s.io/apiextensions-apiserver/pkg/client/informers/externalversions/apiextensions",
    deps = [
        "//vendor/k8s.io/apiextensions-apiserver/pkg/client/informers/externalversions/apiextensions/v1beta1:go_default_library",
        "//vendor/k8s.io/apiextensions-apiserver/pkg/client/informers/externalversions/internalinterfaces:go_default_library",
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
        "//staging/src/k8s.io/apiextensions-apiserver/pkg/client/informers/externalversions/apiextensions/v1beta1:all-srcs",
    ],
    tags = ["automanaged"],
)
