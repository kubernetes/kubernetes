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
    name = "go-to-protobuf",
    srcs = ["main.go"],
    tags = ["automanaged"],
    deps = [
        "//cmd/libs/go2idl/go-to-protobuf/protobuf:go_default_library",
        "//vendor:github.com/spf13/pflag",
    ],
)
