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
    name = "linkcheck",
    srcs = ["links.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/mvdan/xurls",
        "//vendor:github.com/spf13/pflag",
    ],
)
