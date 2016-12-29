package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = ["converter.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/runtime:go_default_library",
        "//pkg/util/json:go_default_library",
    ],
)

go_test(
    name = "go_default_xtest",
    srcs = ["converter_test.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/testapi:go_default_library",
        "//pkg/api/testing:go_default_library",
        "//pkg/runtime:go_default_library",
        "//pkg/runtime/unstructured:go_default_library",
        "//pkg/util/diff:go_default_library",
        "//pkg/util/json:go_default_library",
        "//pkg/util/sets:go_default_library",
        "//vendor:github.com/google/gofuzz",
    ],
)
