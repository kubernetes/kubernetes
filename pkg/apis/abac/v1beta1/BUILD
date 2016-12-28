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
        "conversion.go",
        "doc.go",
        "register.go",
        "types.go",
        "zz_generated.conversion.go",
        "zz_generated.deepcopy.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/apis/abac:go_default_library",
        "//pkg/apis/meta/v1:go_default_library",
        "//pkg/auth/user:go_default_library",
        "//pkg/conversion:go_default_library",
        "//pkg/runtime:go_default_library",
        "//pkg/runtime/schema:go_default_library",
    ],
)

go_test(
    name = "go_default_xtest",
    srcs = ["conversion_test.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/apis/abac:go_default_library",
        "//pkg/apis/abac/v1beta1:go_default_library",
        "//pkg/auth/user:go_default_library",
    ],
)
