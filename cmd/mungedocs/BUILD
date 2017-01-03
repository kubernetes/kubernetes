package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
    "go_test",
)

go_binary(
    name = "mungedocs",
    library = ":go_default_library",
    tags = ["automanaged"],
)

go_test(
    name = "go_default_test",
    srcs = [
        "analytics_test.go",
        "example_syncer_test.go",
        "headers_test.go",
        "kubectl_dash_f_test.go",
        "links_test.go",
        "preformatted_test.go",
        "toc_test.go",
        "util_test.go",
        "whitespace_test.go",
    ],
    data = [
        "mungedocs.go",
        ":testdata",
        "//docs:srcs",
    ],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = ["//vendor:github.com/stretchr/testify/assert"],
)

go_library(
    name = "go_default_library",
    srcs = [
        "analytics.go",
        "example_syncer.go",
        "headers.go",
        "kubectl_dash_f.go",
        "links.go",
        "mungedocs.go",
        "preformatted.go",
        "toc.go",
        "util.go",
        "whitespace.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:github.com/spf13/pflag"],
)

filegroup(
    name = "testdata",
    srcs = glob(["testdata/*"]),
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
