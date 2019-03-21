load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "client.go",
        "signer.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/vmware/govmomi/sts",
    importpath = "github.com/vmware/govmomi/sts",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/google/uuid:go_default_library",
        "//vendor/github.com/vmware/govmomi/lookup:go_default_library",
        "//vendor/github.com/vmware/govmomi/lookup/types:go_default_library",
        "//vendor/github.com/vmware/govmomi/sts/internal:go_default_library",
        "//vendor/github.com/vmware/govmomi/vim25:go_default_library",
        "//vendor/github.com/vmware/govmomi/vim25/methods:go_default_library",
        "//vendor/github.com/vmware/govmomi/vim25/soap:go_default_library",
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
    srcs = [
        ":package-srcs",
        "//vendor/github.com/vmware/govmomi/sts/internal:all-srcs",
        "//vendor/github.com/vmware/govmomi/sts/simulator:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
