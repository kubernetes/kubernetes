package(default_visibility = ["//visibility:public"])

load(
    "//build:go.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "bindata.go",
        "doc.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/generated",
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
        "//pkg/generated/openapi:all-srcs",
    ],
    tags = ["automanaged"],
)
