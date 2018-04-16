package(default_visibility = ["//visibility:public"])

load(
    "//build:go.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = ["kube_features.go"],
    importpath = "k8s.io/apiextensions-apiserver/pkg/features",
    deps = ["//vendor/k8s.io/apiserver/pkg/util/feature:go_default_library"],
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
)
