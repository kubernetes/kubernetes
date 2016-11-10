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
    srcs = ["cgroups_linux.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/opencontainers/runc/libcontainer/cgroups",
        "//vendor:github.com/opencontainers/runc/libcontainer/utils",
    ],
)
