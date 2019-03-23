load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["client.go"],
    importmap = "k8s.io/kubernetes/vendor/github.com/vmware/govmomi/lookup",
    importpath = "github.com/vmware/govmomi/lookup",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/vmware/govmomi/lookup/methods:go_default_library",
        "//vendor/github.com/vmware/govmomi/lookup/types:go_default_library",
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
        "//vendor/github.com/vmware/govmomi/lookup/methods:all-srcs",
        "//vendor/github.com/vmware/govmomi/lookup/simulator:all-srcs",
        "//vendor/github.com/vmware/govmomi/lookup/types:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
