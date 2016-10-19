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
    srcs = ["interfaces.go"],
    tags = ["automanaged"],
    deps = [
        "//federation/pkg/dnsprovider/rrstype:go_default_library",
        "//vendor:google.golang.org/api/googleapi",
    ],
)
