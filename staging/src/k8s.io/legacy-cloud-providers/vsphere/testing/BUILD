load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["testing.go"],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/legacy-cloud-providers/vsphere/testing",
    importpath = "k8s.io/legacy-cloud-providers/vsphere/testing",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/vmware/govmomi/lookup/simulator:go_default_library",
        "//vendor/github.com/vmware/govmomi/simulator:go_default_library",
        "//vendor/github.com/vmware/govmomi/sts/simulator:go_default_library",
        "//vendor/github.com/vmware/govmomi/vapi/simulator:go_default_library",
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
