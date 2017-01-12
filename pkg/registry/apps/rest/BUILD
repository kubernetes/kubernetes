package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = ["storage_apps.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/api/rest:go_default_library",
        "//pkg/apis/apps:go_default_library",
        "//pkg/apis/apps/v1beta1:go_default_library",
        "//pkg/genericapiserver:go_default_library",
        "//pkg/registry/apps/petset/storage:go_default_library",
        "//pkg/registry/generic:go_default_library",
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
