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
    deps = [
        "//pkg/apis/abac:go_default_library",
        "//pkg/apis/abac/latest:go_default_library",
        "//pkg/apis/abac/v0:go_default_library",
        "//pkg/auth/authorizer:go_default_library",
        "//pkg/runtime:go_default_library",
        "//vendor:github.com/golang/glog",
    ],
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
    deps = [
        "//pkg/apis/abac:go_default_library",
        "//pkg/apis/abac/v0:go_default_library",
        "//pkg/apis/abac/v1beta1:go_default_library",
        "//pkg/auth/authorizer:go_default_library",
        "//pkg/auth/user:go_default_library",
        "//pkg/runtime:go_default_library",
    ],
)
