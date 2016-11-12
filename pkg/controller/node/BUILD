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
        "cidr_allocator.go",
        "cidr_set.go",
        "controller_utils.go",
        "doc.go",
        "metrics.go",
        "nodecontroller.go",
        "rate_limited_queue.go",
        "test_utils.go",
    ],
    tags = ["automanaged"],
)

go_test(
    name = "go_default_test",
    srcs = [
        "cidr_allocator_test.go",
        "cidr_set_test.go",
        "nodecontroller_test.go",
        "rate_limited_queue_test.go",
    ],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [],
)
