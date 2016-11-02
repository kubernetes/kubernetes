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
    name = "client-gen",
    srcs = ["main.go"],
    tags = ["automanaged"],
    deps = [
        "//cmd/libs/go2idl/client-gen/args:go_default_library",
        "//cmd/libs/go2idl/client-gen/generators:go_default_library",
        "//cmd/libs/go2idl/client-gen/types:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/spf13/pflag",
        "//vendor:k8s.io/gengo/args",
    ],
)
