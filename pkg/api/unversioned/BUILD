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
    srcs = [
        "duration.go",
        "time.go",
        "types.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/genericapiserver/openapi/common:go_default_library",
        "//vendor:github.com/go-openapi/spec",
        "//vendor:github.com/google/gofuzz",
    ],
)
