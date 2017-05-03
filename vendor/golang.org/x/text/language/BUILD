package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "common.go",
        "coverage.go",
        "go1_2.go",
        "index.go",
        "language.go",
        "lookup.go",
        "match.go",
        "parse.go",
        "tables.go",
        "tags.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor/golang.org/x/text/internal/tag:go_default_library"],
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
