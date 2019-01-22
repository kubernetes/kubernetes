package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)
load("//build:bindata.bzl", "go_bindata")

# IMPORTANT: if you make any changes here, you must also update hack/generate-bindata.sh.
go_bindata(
    name = "bindata",
    srcs = [
        "//translations:all-srcs",
    ],
    outs = ["bindata.go"],
    compress = False,
    include_metadata = False,
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
