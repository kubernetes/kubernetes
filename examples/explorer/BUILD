package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
)

go_binary(
    name = "explorer",
    srcs = ["explorer.go"],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/davecgh/go-spew/spew"],
)
