load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "delegate.go",
        "doc.go",
        "requests.go",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/rackspace/gophercloud:go_default_library",
        "//vendor/github.com/rackspace/gophercloud/openstack/compute/v2/extensions/bootfromvolume:go_default_library",
        "//vendor/github.com/rackspace/gophercloud/openstack/compute/v2/extensions/diskconfig:go_default_library",
        "//vendor/github.com/rackspace/gophercloud/openstack/compute/v2/servers:go_default_library",
        "//vendor/github.com/rackspace/gophercloud/pagination:go_default_library",
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
    visibility = ["//visibility:public"],
)
