load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "asymmetric.go",
        "crypter.go",
        "doc.go",
        "encoding.go",
        "jwe.go",
        "jwk.go",
        "jws.go",
        "shared.go",
        "signing.go",
        "symmetric.go",
        "utils.go",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/square/go-jose/cipher:go_default_library",
        "//vendor/github.com/square/go-jose/json:go_default_library",
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
        "//vendor/github.com/square/go-jose/cipher:all-srcs",
        "//vendor/github.com/square/go-jose/json:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
