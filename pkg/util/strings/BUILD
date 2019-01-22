package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "escape.go",
        "line_delimiter.go",
        "strings.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/util/strings",
)

go_test(
    name = "go_default_test",
    srcs = [
        "escape_test.go",
        "line_delimiter_test.go",
        "strings_test.go",
    ],
    embed = [":go_default_library"],
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
