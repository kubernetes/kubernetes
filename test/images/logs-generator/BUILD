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
    name = "logs-generator",
    srcs = ["logs_generator.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/util/rand:go_default_library",
        "//vendor:github.com/golang/glog",
    ],
)
