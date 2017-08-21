load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "auth_env.go",
        "client.go",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/rackspace/gophercloud:go_default_library",
        "//vendor/github.com/rackspace/gophercloud/openstack:go_default_library",
        "//vendor/github.com/rackspace/gophercloud/openstack/utils:go_default_library",
        "//vendor/github.com/rackspace/gophercloud/rackspace/identity/v2/tokens:go_default_library",
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
    srcs = [
        ":package-srcs",
        "//vendor/github.com/rackspace/gophercloud/rackspace/blockstorage/v1/volumes:all-srcs",
        "//vendor/github.com/rackspace/gophercloud/rackspace/compute/v2/servers:all-srcs",
        "//vendor/github.com/rackspace/gophercloud/rackspace/compute/v2/volumeattach:all-srcs",
        "//vendor/github.com/rackspace/gophercloud/rackspace/identity/v2/tokens:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
