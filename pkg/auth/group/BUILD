package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
    "go_test",
    "cgo_library",
)

go_library(
    name = "go_default_library",
    srcs = ["group_adder.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/auth/authenticator:go_default_library",
        "//pkg/auth/user:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["group_adder_test.go"],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/auth/authenticator:go_default_library",
        "//pkg/auth/user:go_default_library",
    ],
)
