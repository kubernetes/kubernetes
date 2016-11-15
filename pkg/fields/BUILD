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
        "fields.go",
        "requirements.go",
        "selector.go",
    ],
    tags = ["automanaged"],
    deps = ["//pkg/selection:go_default_library"],
)

go_test(
    name = "go_default_test",
    srcs = [
        "fields_test.go",
        "selector_test.go",
    ],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [],
)
