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
    name = "local",
    srcs = ["run_local.go"],
    tags = ["automanaged"],
    deps = [
        "//test/e2e_node/builder:go_default_library",
        "//vendor:github.com/golang/glog",
    ],
)
