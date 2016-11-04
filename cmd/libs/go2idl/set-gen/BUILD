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
    name = "set-gen",
    srcs = ["main.go"],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/golang/glog"],
)

exports_files([
    "types.go",
])
