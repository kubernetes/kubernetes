package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = ["doc.go"],
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
    srcs = [
        ":package-srcs",
        "//plugin/pkg/auth/authenticator/password:all-srcs",
        "//plugin/pkg/auth/authenticator/request/basicauth:all-srcs",
        "//plugin/pkg/auth/authenticator/token/anytoken:all-srcs",
        "//plugin/pkg/auth/authenticator/token/oidc:all-srcs",
        "//plugin/pkg/auth/authenticator/token/tokentest:all-srcs",
        "//plugin/pkg/auth/authenticator/token/webhook:all-srcs",
    ],
    tags = ["automanaged"],
)
