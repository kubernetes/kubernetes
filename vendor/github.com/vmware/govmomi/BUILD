load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["client.go"],
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/vmware/govmomi/property:go_default_library",
        "//vendor/github.com/vmware/govmomi/session:go_default_library",
        "//vendor/github.com/vmware/govmomi/vim25:go_default_library",
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
        "//vendor/github.com/vmware/govmomi/find:all-srcs",
        "//vendor/github.com/vmware/govmomi/list:all-srcs",
        "//vendor/github.com/vmware/govmomi/object:all-srcs",
        "//vendor/github.com/vmware/govmomi/pbm:all-srcs",
        "//vendor/github.com/vmware/govmomi/property:all-srcs",
        "//vendor/github.com/vmware/govmomi/session:all-srcs",
        "//vendor/github.com/vmware/govmomi/task:all-srcs",
        "//vendor/github.com/vmware/govmomi/vim25:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
