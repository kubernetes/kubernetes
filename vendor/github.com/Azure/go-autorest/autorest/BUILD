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
        "retriablerequest.go",
        "retriablerequest_1.7.go",
        "retriablerequest_1.8.go",
        "sender.go",
        "utility.go",
        "version.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/Azure/go-autorest/autorest",
    importpath = "github.com/Azure/go-autorest/autorest",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/Azure/go-autorest/autorest/adal:go_default_library",
        "//vendor/github.com/Azure/go-autorest/logger:go_default_library",
        "//vendor/github.com/Azure/go-autorest/tracing:go_default_library",
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
        "//vendor/github.com/Azure/go-autorest/autorest/adal:all-srcs",
        "//vendor/github.com/Azure/go-autorest/autorest/azure:all-srcs",
        "//vendor/github.com/Azure/go-autorest/autorest/date:all-srcs",
        "//vendor/github.com/Azure/go-autorest/autorest/to:all-srcs",
        "//vendor/github.com/Azure/go-autorest/autorest/validation:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
