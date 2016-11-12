package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
    "go_test",
    "cgo_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "aws.go",
        "aws_instancegroups.go",
        "aws_loadbalancer.go",
        "aws_routes.go",
        "aws_utils.go",
        "log_handler.go",
        "retry_handler.go",
        "sets_ippermissions.go",
        "volumes.go",
    ],
    tags = ["automanaged"],
)

go_test(
    name = "go_default_test",
    srcs = [
        "aws_test.go",
        "retry_handler_test.go",
    ],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [],
)
