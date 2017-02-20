package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "aws.go",
        "aws_instancegroups.go",
        "aws_loadbalancer.go",
        "aws_routes.go",
        "aws_utils.go",
        "device_allocator.go",
        "log_handler.go",
        "regions.go",
        "retry_handler.go",
        "sets_ippermissions.go",
        "volumes.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api/v1/service:go_default_library",
        "//pkg/cloudprovider:go_default_library",
        "//pkg/credentialprovider/aws:go_default_library",
        "//pkg/volume:go_default_library",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "aws_test.go",
        "device_allocator_test.go",
        "regions_test.go",
        "retry_handler_test.go",
    ],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = ["//vendor:k8s.io/apimachinery/pkg/apis/meta/v1"],
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
)
