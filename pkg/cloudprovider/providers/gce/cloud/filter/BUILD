load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = ["filter.go"],
    importpath = "k8s.io/kubernetes/pkg/cloudprovider/providers/gce/cloud/filter",
    visibility = ["//visibility:public"],
    deps = ["//vendor/github.com/golang/glog:go_default_library"],
)

go_test(
    name = "go_default_test",
    srcs = ["filter_test.go"],
    embed = [":go_default_library"],
    importpath = "k8s.io/kubernetes/pkg/cloudprovider/providers/gce/cloud/filter",
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
