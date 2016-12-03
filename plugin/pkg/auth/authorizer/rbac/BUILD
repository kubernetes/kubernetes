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
        "rbac.go",
        "subject_locator.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/apis/rbac:go_default_library",
        "//pkg/apis/rbac/validation:go_default_library",
        "//pkg/auth/authorizer:go_default_library",
        "//pkg/auth/user:go_default_library",
        "//pkg/util/errors:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "rbac_test.go",
        "subject_locator_test.go",
    ],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/apis/rbac:go_default_library",
        "//pkg/apis/rbac/validation:go_default_library",
        "//pkg/auth/authorizer:go_default_library",
        "//pkg/auth/user:go_default_library",
    ],
)
