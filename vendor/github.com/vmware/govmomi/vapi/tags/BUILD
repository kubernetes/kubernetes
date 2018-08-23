load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "categories.go",
        "tag_association.go",
        "tags.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/vmware/govmomi/vapi/tags",
    importpath = "github.com/vmware/govmomi/vapi/tags",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/vmware/govmomi/vapi/internal:go_default_library",
        "//vendor/github.com/vmware/govmomi/vapi/rest:go_default_library",
        "//vendor/github.com/vmware/govmomi/vim25/mo:go_default_library",
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
