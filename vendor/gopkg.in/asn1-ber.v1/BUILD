load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "ber.go",
        "content_int.go",
        "header.go",
        "identifier.go",
        "length.go",
        "util.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/gopkg.in/asn1-ber.v1",
    importpath = "gopkg.in/asn1-ber.v1",
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
