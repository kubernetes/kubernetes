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
    srcs = ["client.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/apimachinery/registered:go_default_library",
        "//pkg/client/restclient:go_default_library",
        "//pkg/client/typed/dynamic:go_default_library",
    ],
)
