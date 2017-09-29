load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "client.go",
        "doc.go",
        "retry.go",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/vmware/govmomi/vim25/methods:go_default_library",
        "//vendor/github.com/vmware/govmomi/vim25/soap:go_default_library",
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
    srcs = [
        ":package-srcs",
        "//vendor/github.com/vmware/govmomi/vim25/debug:all-srcs",
        "//vendor/github.com/vmware/govmomi/vim25/methods:all-srcs",
        "//vendor/github.com/vmware/govmomi/vim25/mo:all-srcs",
        "//vendor/github.com/vmware/govmomi/vim25/progress:all-srcs",
        "//vendor/github.com/vmware/govmomi/vim25/soap:all-srcs",
        "//vendor/github.com/vmware/govmomi/vim25/types:all-srcs",
        "//vendor/github.com/vmware/govmomi/vim25/xml:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
