package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
    "go_test",
    "cgo_library",
)

go_binary(
    name = "clicheck",
    srcs = ["check_cli_conventions.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/kubectl/cmd:go_default_library",
        "//pkg/kubectl/cmd/util:go_default_library",
        "//pkg/kubectl/cmd/util/sanity:go_default_library",
    ],
)
