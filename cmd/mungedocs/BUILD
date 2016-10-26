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
    name = "mungedocs",
    srcs = [
        "analytics.go",
        "example_syncer.go",
        "headers.go",
        "kubectl_dash_f.go",
        "links.go",
        "mungedocs.go",
        "preformatted.go",
        "toc.go",
        "util.go",
        "whitespace.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/spf13/pflag"],
)
