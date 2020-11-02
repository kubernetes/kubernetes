load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "client.go",
        "debug.go",
        "error.go",
        "soap.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/vmware/govmomi/vim25/soap",
    importpath = "github.com/vmware/govmomi/vim25/soap",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/vmware/govmomi/vim25/debug:go_default_library",
        "//vendor/github.com/vmware/govmomi/vim25/progress:go_default_library",
        "//vendor/github.com/vmware/govmomi/vim25/types:go_default_library",
        "//vendor/github.com/vmware/govmomi/vim25/xml:go_default_library",
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
