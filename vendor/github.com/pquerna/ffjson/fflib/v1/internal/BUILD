load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "atof.go",
        "atoi.go",
        "extfloat.go",
        "ftoa.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/pquerna/ffjson/fflib/v1/internal",
    importpath = "github.com/pquerna/ffjson/fflib/v1/internal",
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
