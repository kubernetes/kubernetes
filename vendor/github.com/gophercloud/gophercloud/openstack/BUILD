load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "auth_env.go",
        "client.go",
        "doc.go",
        "endpoint_location.go",
        "errors.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/gophercloud/gophercloud/openstack",
    importpath = "github.com/gophercloud/gophercloud/openstack",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/gophercloud/gophercloud:go_default_library",
        "//vendor/github.com/gophercloud/gophercloud/openstack/identity/v2/tokens:go_default_library",
        "//vendor/github.com/gophercloud/gophercloud/openstack/identity/v3/tokens:go_default_library",
        "//vendor/github.com/gophercloud/gophercloud/openstack/utils:go_default_library",
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
        "//vendor/github.com/gophercloud/gophercloud/openstack/blockstorage/extensions/volumeactions:all-srcs",
        "//vendor/github.com/gophercloud/gophercloud/openstack/blockstorage/v1/volumes:all-srcs",
        "//vendor/github.com/gophercloud/gophercloud/openstack/blockstorage/v2/volumes:all-srcs",
        "//vendor/github.com/gophercloud/gophercloud/openstack/blockstorage/v3/volumes:all-srcs",
        "//vendor/github.com/gophercloud/gophercloud/openstack/common/extensions:all-srcs",
        "//vendor/github.com/gophercloud/gophercloud/openstack/compute/v2/extensions/attachinterfaces:all-srcs",
        "//vendor/github.com/gophercloud/gophercloud/openstack/compute/v2/extensions/volumeattach:all-srcs",
        "//vendor/github.com/gophercloud/gophercloud/openstack/compute/v2/flavors:all-srcs",
        "//vendor/github.com/gophercloud/gophercloud/openstack/compute/v2/images:all-srcs",
        "//vendor/github.com/gophercloud/gophercloud/openstack/compute/v2/servers:all-srcs",
        "//vendor/github.com/gophercloud/gophercloud/openstack/identity/v2/tenants:all-srcs",
        "//vendor/github.com/gophercloud/gophercloud/openstack/identity/v2/tokens:all-srcs",
        "//vendor/github.com/gophercloud/gophercloud/openstack/identity/v3/extensions/trusts:all-srcs",
        "//vendor/github.com/gophercloud/gophercloud/openstack/identity/v3/tokens:all-srcs",
        "//vendor/github.com/gophercloud/gophercloud/openstack/networking/v2/extensions:all-srcs",
        "//vendor/github.com/gophercloud/gophercloud/openstack/networking/v2/networks:all-srcs",
        "//vendor/github.com/gophercloud/gophercloud/openstack/networking/v2/ports:all-srcs",
        "//vendor/github.com/gophercloud/gophercloud/openstack/utils:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
