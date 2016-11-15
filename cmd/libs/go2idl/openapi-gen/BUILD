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
    name = "openapi-gen",
    srcs = ["main.go"],
    tags = ["automanaged"],
    deps = [
        "//cmd/libs/go2idl/openapi-gen/generators:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:k8s.io/gengo/args",
    ],
)
