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
    name = "import-boss",
    srcs = ["main.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/golang/glog",
        "//vendor:k8s.io/gengo/args",
        "//vendor:k8s.io/gengo/examples/import-boss/generators",
    ],
)
