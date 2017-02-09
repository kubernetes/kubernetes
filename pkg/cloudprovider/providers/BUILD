package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
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
        "//pkg/cloudprovider/providers/aws:all-srcs",
        "//pkg/cloudprovider/providers/azure:all-srcs",
        "//pkg/cloudprovider/providers/cloudstack:all-srcs",
        "//pkg/cloudprovider/providers/fake:all-srcs",
        "//pkg/cloudprovider/providers/gce:all-srcs",
        "//pkg/cloudprovider/providers/mesos:all-srcs",
        "//pkg/cloudprovider/providers/openstack:all-srcs",
        "//pkg/cloudprovider/providers/ovirt:all-srcs",
        "//pkg/cloudprovider/providers/photon:all-srcs",
        "//pkg/cloudprovider/providers/rackspace:all-srcs",
        "//pkg/cloudprovider/providers/vsphere:all-srcs",
    ],
    tags = ["automanaged"],
)
