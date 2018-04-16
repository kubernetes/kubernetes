package(default_visibility = ["//visibility:public"])

load(
    "//build:go.bzl",
    "go_binary",
    "go_library",
)

go_binary(
    name = "nonewprivs",
    embed = [":go_default_library"],
)

go_library(
    name = "go_default_library",
    srcs = ["nnp.go"],
    importpath = "k8s.io/kubernetes/test/images/nonewprivs",
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
