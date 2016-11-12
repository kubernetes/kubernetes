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
        "doc.go",
        "replenishment_controller.go",
        "resource_quota_controller.go",
    ],
    tags = ["automanaged"],
)

go_test(
    name = "go_default_test",
    srcs = [
        "replenishment_controller_test.go",
        "resource_quota_controller_test.go",
    ],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [],
)
