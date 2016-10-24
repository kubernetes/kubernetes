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
        "escape.go",
        "line_delimiter.go",
        "strings.go",
    ],
    tags = ["automanaged"],
)

go_test(
    name = "go_default_test",
    srcs = [
        "escape_test.go",
        "line_delimiter_test.go",
        "strings_test.go",
    ],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [],
)
