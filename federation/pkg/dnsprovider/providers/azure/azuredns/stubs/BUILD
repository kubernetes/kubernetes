load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["azurednsapi.go"],
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/Azure/azure-sdk-for-go/arm/dns:go_default_library",
        "//vendor/github.com/Azure/go-autorest/autorest:go_default_library",
        "//vendor/github.com/Azure/go-autorest/autorest/to:go_default_library",
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
    srcs = [":package-srcs"],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
