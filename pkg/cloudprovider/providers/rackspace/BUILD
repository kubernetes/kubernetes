package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
    "go_test",
    "cgo_library",
)

go_library(
    name = "go_default_library",
    srcs = ["rackspace.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/api/v1:go_default_library",
        "//pkg/cloudprovider:go_default_library",
        "//pkg/types:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/rackspace/gophercloud",
        "//vendor:github.com/rackspace/gophercloud/openstack/compute/v2/extensions/volumeattach",
        "//vendor:github.com/rackspace/gophercloud/openstack/compute/v2/servers",
        "//vendor:github.com/rackspace/gophercloud/pagination",
        "//vendor:github.com/rackspace/gophercloud/rackspace",
        "//vendor:github.com/rackspace/gophercloud/rackspace/blockstorage/v1/volumes",
        "//vendor:github.com/rackspace/gophercloud/rackspace/compute/v2/servers",
        "//vendor:github.com/rackspace/gophercloud/rackspace/compute/v2/volumeattach",
        "//vendor:gopkg.in/gcfg.v1",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["rackspace_test.go"],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = ["//vendor:github.com/rackspace/gophercloud"],
)
