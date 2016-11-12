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
    srcs = [
        "metadata.go",
        "openstack.go",
        "openstack_instances.go",
        "openstack_loadbalancer.go",
        "openstack_volumes.go",
    ],
    tags = ["automanaged"],
)

go_test(
    name = "go_default_test",
    srcs = [
        "metadata_test.go",
        "openstack_test.go",
    ],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [],
)
