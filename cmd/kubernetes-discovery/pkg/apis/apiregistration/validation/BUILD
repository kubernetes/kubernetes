package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = ["validation.go"],
    tags = ["automanaged"],
    deps = [
        "//cmd/kubernetes-discovery/pkg/apis/apiregistration:go_default_library",
        "//pkg/api/validation:go_default_library",
        "//pkg/api/validation/path:go_default_library",
        "//pkg/util/validation:go_default_library",
        "//pkg/util/validation/field:go_default_library",
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
    srcs = [":package-srcs"],
    tags = ["automanaged"],
)
