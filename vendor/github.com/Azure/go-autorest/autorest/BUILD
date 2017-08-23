load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "authorization.go",
        "autorest.go",
        "client.go",
        "error.go",
        "preparer.go",
        "responder.go",
        "sender.go",
        "utility.go",
        "version.go",
    ],
    visibility = ["//visibility:public"],
    deps = ["//vendor/github.com/Azure/go-autorest/autorest/adal:go_default_library"],
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
        "//vendor/github.com/Azure/go-autorest/autorest/adal:all-srcs",
        "//vendor/github.com/Azure/go-autorest/autorest/azure:all-srcs",
        "//vendor/github.com/Azure/go-autorest/autorest/date:all-srcs",
        "//vendor/github.com/Azure/go-autorest/autorest/to:all-srcs",
        "//vendor/github.com/Azure/go-autorest/autorest/validation:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
