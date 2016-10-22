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
    name = "genswaggertypedocs",
    srcs = ["swagger_type_docs.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/runtime:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/spf13/pflag",
    ],
)
