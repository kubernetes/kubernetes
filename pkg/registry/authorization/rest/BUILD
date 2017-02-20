package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = ["storage_authorization.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/registry/authorization/localsubjectaccessreview:go_default_library",
        "//pkg/registry/authorization/selfsubjectaccessreview:go_default_library",
        "//pkg/registry/authorization/subjectaccessreview:go_default_library",
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
