load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = ["cmdutil.go"],
    visibility = ["//visibility:public"],
    deps = ["//vendor/github.com/spf13/cobra:go_default_library"],
)

go_test(
    name = "go_default_test",
    srcs = ["cmdutil_test.go"],
    library = ":go_default_library",
)

filegroup(
    name = "package-srcs",
    srcs = glob(["**"]),
    tags = ["automanaged"],
    visibility = ["//visibility:private"],
)

filegroup(
    name = "all-srcs",
    srcs = [":package-srcs"],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
