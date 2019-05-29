package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
)

go_binary(
    name = "resource-consumer",
    embed = [":go_default_library"],
)

go_library(
    name = "go_default_library",
    srcs = [
        "resource_consumer.go",
        "resource_consumer_handler.go",
        "utils.go",
    ],
    importpath = "k8s.io/kubernetes/test/images/resource-consumer",
    deps = ["//test/images/resource-consumer/common:go_default_library"],
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
        "//test/images/resource-consumer/common:all-srcs",
        "//test/images/resource-consumer/consume-cpu:all-srcs",
    ],
    tags = ["automanaged"],
)
