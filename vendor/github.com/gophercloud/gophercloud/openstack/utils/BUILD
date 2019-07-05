load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "base_endpoint.go",
        "choose_version.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/gophercloud/gophercloud/openstack/utils",
    importpath = "github.com/gophercloud/gophercloud/openstack/utils",
    visibility = ["//visibility:public"],
    deps = ["//vendor/github.com/gophercloud/gophercloud:go_default_library"],
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
