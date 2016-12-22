package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
)

go_binary(
    name = "server",
    srcs = ["main.go"],
    tags = ["automanaged"],
    deps = [
        "//examples/apiserver:go_default_library",
        "//pkg/util/flag:go_default_library",
        "//pkg/util/wait:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/spf13/pflag",
    ],
)
