package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "keystone.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/util/cert:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/rackspace/gophercloud",
        "//vendor:github.com/rackspace/gophercloud/openstack",
        "//vendor:k8s.io/apimachinery/pkg/util/net",
        "//vendor:k8s.io/apiserver/pkg/authentication/user",
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
)
