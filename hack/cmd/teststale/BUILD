package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
    "go_test",
)

go_binary(
    name = "teststale",
    library = ":go_default_library",
)

go_test(
    name = "go_default_test",
    srcs = ["teststale_test.go"],
    library = ":go_default_library",
)

go_library(
    name = "go_default_library",
    srcs = ["teststale.go"],
    deps = ["//vendor/github.com/golang/glog:go_default_library"],
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
)
