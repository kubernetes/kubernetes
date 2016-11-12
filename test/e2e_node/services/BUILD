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
        "apiserver.go",
        "etcd.go",
        "internal_services.go",
        "namespace_controller.go",
        "server.go",
        "services.go",
    ],
    tags = ["automanaged"],
)
