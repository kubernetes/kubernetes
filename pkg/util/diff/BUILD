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
    srcs = ["diff.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/util/validation/field:go_default_library",
        "//vendor:github.com/davecgh/go-spew/spew",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["diff_test.go"],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [],
)
