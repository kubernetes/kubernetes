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
        "conversion.go",
        "register.go",
        "types.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api/unversioned:go_default_library",
        "//pkg/apis/abac:go_default_library",
        "//pkg/conversion:go_default_library",
        "//pkg/runtime:go_default_library",
    ],
)

go_test(
    name = "go_default_xtest",
    srcs = ["conversion_test.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/apis/abac:go_default_library",
        "//pkg/apis/abac/v0:go_default_library",
    ],
)
