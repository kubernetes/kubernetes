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
        "etcd_utils.go",
        "master_utils.go",
        "perf_utils.go",
        "serializer.go",
    ],
    tags = ["automanaged"],
)
