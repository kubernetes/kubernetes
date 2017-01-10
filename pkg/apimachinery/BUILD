package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "types.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api/meta:go_default_library",
        "//pkg/runtime:go_default_library",
        "//pkg/runtime/schema:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["types_test.go"],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = ["//pkg/runtime/schema:go_default_library"],
)

filegroup(
    name = "package-srcs",
    srcs = glob(["**"]),
    tags = ["automanaged"],
    visibility = ["//visibility:private"],
)

filegroup(
    name = "all-srcs",
    srcs = [
        ":package-srcs",
        "//pkg/apimachinery/announced:all-srcs",
        "//pkg/apimachinery/registered:all-srcs",
        "//pkg/apimachinery/tests:all-srcs",
    ],
    tags = ["automanaged"],
)
