load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "accessors.go",
        "constants.go",
        "conversions.go",
        "doc.go",
        "map.go",
        "mutations.go",
        "security.go",
        "tests.go",
        "type_specific_codegen.go",
        "value.go",
    ],
    visibility = ["//visibility:public"],
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
