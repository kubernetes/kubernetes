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
        "//plugin/pkg/auth/authenticator/password/allow:all-srcs",
        "//plugin/pkg/auth/authenticator/password/keystone:all-srcs",
        "//plugin/pkg/auth/authenticator/password/passwordfile:all-srcs",
    ],
    tags = ["automanaged"],
)
