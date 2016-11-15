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
    name = "kubefed",
    srcs = ["kubefed.go"],
    tags = ["automanaged"],
    deps = ["//federation/cmd/kubefed/app:go_default_library"],
)
