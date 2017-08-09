package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = ["plugins.go"],
    deps = ["//plugin/pkg/scheduler/algorithmprovider/defaults:go_default_library"],
)

go_test(
    name = "go_default_test",
    srcs = ["plugins_test.go"],
    library = ":go_default_library",
    deps = ["//plugin/pkg/scheduler/factory:go_default_library"],
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
        "//plugin/pkg/scheduler/algorithmprovider/defaults:all-srcs",
    ],
    tags = ["automanaged"],
)
