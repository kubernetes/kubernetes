load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "config.go",
        "devicetoken.go",
        "persist.go",
        "sender.go",
        "token.go",
        "version.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/Azure/go-autorest/autorest/adal",
    importpath = "github.com/Azure/go-autorest/autorest/adal",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/Azure/go-autorest/autorest/date:go_default_library",
        "//vendor/github.com/Azure/go-autorest/tracing:go_default_library",
        "//vendor/github.com/dgrijalva/jwt-go:go_default_library",
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
