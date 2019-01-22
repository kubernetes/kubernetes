load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["types.go"],
    importmap = "k8s.io/kubernetes/vendor/github.com/vmware/govmomi/sts/internal",
    importpath = "github.com/vmware/govmomi/sts/internal",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/vmware/govmomi/vim25/soap:go_default_library",
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
