load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["helpers.go"],
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/cloudflare/cfssl/crypto/pkcs7:go_default_library",
        "//vendor/github.com/cloudflare/cfssl/errors:go_default_library",
        "//vendor/github.com/cloudflare/cfssl/helpers/derhelpers:go_default_library",
        "//vendor/github.com/cloudflare/cfssl/log:go_default_library",
        "//vendor/golang.org/x/crypto/pkcs12:go_default_library",
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
        "//vendor/github.com/cloudflare/cfssl/helpers/derhelpers:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
