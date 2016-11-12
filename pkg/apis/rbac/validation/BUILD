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
        "policy_comparator.go",
        "rulevalidation.go",
        "validation.go",
    ],
    tags = ["automanaged"],
)

go_test(
    name = "go_default_test",
    srcs = [
        "policy_comparator_test.go",
        "rulevalidation_test.go",
        "validation_test.go",
    ],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [],
)
