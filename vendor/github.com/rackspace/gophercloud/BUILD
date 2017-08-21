load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "auth_options.go",
        "auth_results.go",
        "doc.go",
        "endpoint_search.go",
        "params.go",
        "provider_client.go",
        "results.go",
        "service_client.go",
        "util.go",
    ],
    visibility = ["//visibility:public"],
    deps = ["//vendor/github.com/mitchellh/mapstructure:go_default_library"],
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
        "//vendor/github.com/rackspace/gophercloud/openstack:all-srcs",
        "//vendor/github.com/rackspace/gophercloud/pagination:all-srcs",
        "//vendor/github.com/rackspace/gophercloud/rackspace:all-srcs",
        "//vendor/github.com/rackspace/gophercloud/testhelper:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
