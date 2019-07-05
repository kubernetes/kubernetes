load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "error.go",
        "finder.go",
        "recurser.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/vmware/govmomi/find",
    importpath = "github.com/vmware/govmomi/find",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/vmware/govmomi/list:go_default_library",
        "//vendor/github.com/vmware/govmomi/object:go_default_library",
        "//vendor/github.com/vmware/govmomi/property:go_default_library",
        "//vendor/github.com/vmware/govmomi/vim25:go_default_library",
        "//vendor/github.com/vmware/govmomi/vim25/mo:go_default_library",
        "//vendor/github.com/vmware/govmomi/vim25/types:go_default_library",
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
