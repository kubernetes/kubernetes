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
    srcs = ["abac.go"],
    tags = ["automanaged"],
)

filegroup(
    name = "example_policy",
    testonly = True,
    srcs = [
        "example_policy_file.jsonl",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["abac_test.go"],
    data = [
        ":example_policy",
    ],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [],
)
