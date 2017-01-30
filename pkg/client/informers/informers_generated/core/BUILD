package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = ["interface.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/client/informers/informers_generated/core/internalversion:go_default_library",
        "//pkg/client/informers/informers_generated/core/v1:go_default_library",
        "//pkg/client/informers/informers_generated/internalinterfaces:go_default_library",
    ],
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
        "//pkg/client/informers/informers_generated/core/internalversion:all-srcs",
        "//pkg/client/informers/informers_generated/core/v1:all-srcs",
    ],
    tags = ["automanaged"],
)
