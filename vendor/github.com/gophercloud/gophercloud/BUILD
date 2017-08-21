load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "auth_options.go",
        "doc.go",
        "endpoint_search.go",
        "errors.go",
        "params.go",
        "provider_client.go",
        "results.go",
        "service_client.go",
        "util.go",
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
    srcs = [
        ":package-srcs",
        "//vendor/github.com/gophercloud/gophercloud/openstack:all-srcs",
        "//vendor/github.com/gophercloud/gophercloud/pagination:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
