package(default_visibility = ["//visibility:public"])

load(
    "//build:go.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = ["interrupt.go"],
    importpath = "k8s.io/kubernetes/pkg/util/interrupt",
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
