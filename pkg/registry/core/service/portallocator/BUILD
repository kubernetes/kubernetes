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
        "allocator.go",
        "operation.go",
    ],
    tags = ["automanaged"],
    deps = ["//pkg/registry/core/service/allocator:go_default_library"],
)

go_test(
    name = "go_default_test",
    srcs = ["allocator_test.go"],
    library = ":go_default_library",
    tags = ["automanaged"],
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
        "//pkg/registry/core/service/portallocator/controller:all-srcs",
    ],
    tags = ["automanaged"],
)
