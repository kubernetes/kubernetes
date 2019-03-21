load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "extras.go",
        "marshal.go",
        "read.go",
        "typeinfo.go",
        "xml.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/vmware/govmomi/vim25/xml",
    importpath = "github.com/vmware/govmomi/vim25/xml",
    visibility = ["//visibility:public"],
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
