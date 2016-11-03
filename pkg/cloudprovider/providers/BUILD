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
    srcs = ["providers.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/cloudprovider/providers/aws:go_default_library",
        "//pkg/cloudprovider/providers/azure:go_default_library",
        "//pkg/cloudprovider/providers/cloudstack:go_default_library",
        "//pkg/cloudprovider/providers/gce:go_default_library",
        "//pkg/cloudprovider/providers/mesos:go_default_library",
        "//pkg/cloudprovider/providers/openstack:go_default_library",
        "//pkg/cloudprovider/providers/ovirt:go_default_library",
        "//pkg/cloudprovider/providers/photon:go_default_library",
        "//pkg/cloudprovider/providers/rackspace:go_default_library",
        "//pkg/cloudprovider/providers/vsphere:go_default_library",
    ],
)
