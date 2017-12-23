package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = ["framer_test.go"],
    embed = [":go_default_library"],
    importpath = "k8s.io/apimachinery/pkg/util/framer",
)

go_library(
    name = "go_default_library",
    srcs = ["framer.go"],
    importpath = "k8s.io/apimachinery/pkg/util/framer",
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
