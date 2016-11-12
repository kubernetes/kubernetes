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
        "doc.go",
        "proxy.go",
        "response_checker.go",
        "streamer.go",
    ],
    tags = ["automanaged"],
)

go_test(
    name = "go_default_test",
    srcs = [
        "proxy_test.go",
        "response_checker_test.go",
        "streamer_test.go",
    ],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [],
)
