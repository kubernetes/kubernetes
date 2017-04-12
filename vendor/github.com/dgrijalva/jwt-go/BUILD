package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "claims.go",
        "doc.go",
        "ecdsa.go",
        "ecdsa_utils.go",
        "errors.go",
        "hmac.go",
        "map_claims.go",
        "none.go",
        "parser.go",
        "rsa.go",
        "rsa_pss.go",
        "rsa_utils.go",
        "signing_method.go",
        "token.go",
    ],
    tags = ["automanaged"],
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
)
