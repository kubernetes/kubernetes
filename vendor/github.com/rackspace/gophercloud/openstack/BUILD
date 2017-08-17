load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "auth_env.go",
        "client.go",
        "endpoint_location.go",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/rackspace/gophercloud:go_default_library",
        "//vendor/github.com/rackspace/gophercloud/openstack/identity/v2/tokens:go_default_library",
        "//vendor/github.com/rackspace/gophercloud/openstack/identity/v3/tokens:go_default_library",
        "//vendor/github.com/rackspace/gophercloud/openstack/utils:go_default_library",
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
        "//vendor/github.com/rackspace/gophercloud/openstack/blockstorage/v1/volumes:all-srcs",
        "//vendor/github.com/rackspace/gophercloud/openstack/compute/v2/extensions/bootfromvolume:all-srcs",
        "//vendor/github.com/rackspace/gophercloud/openstack/compute/v2/extensions/diskconfig:all-srcs",
        "//vendor/github.com/rackspace/gophercloud/openstack/compute/v2/extensions/volumeattach:all-srcs",
        "//vendor/github.com/rackspace/gophercloud/openstack/compute/v2/flavors:all-srcs",
        "//vendor/github.com/rackspace/gophercloud/openstack/compute/v2/images:all-srcs",
        "//vendor/github.com/rackspace/gophercloud/openstack/compute/v2/servers:all-srcs",
        "//vendor/github.com/rackspace/gophercloud/openstack/identity/v2/tenants:all-srcs",
        "//vendor/github.com/rackspace/gophercloud/openstack/identity/v2/tokens:all-srcs",
        "//vendor/github.com/rackspace/gophercloud/openstack/identity/v3/tokens:all-srcs",
        "//vendor/github.com/rackspace/gophercloud/openstack/utils:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
