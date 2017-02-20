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
        "rbac.go",
        "subject_locator.go",
    ],
    tags = ["automanaged"],
    deps = ["//pkg/registry/rbac/validation:go_default_library"],
)

go_test(
    name = "go_default_test",
    srcs = [
        "rbac_test.go",
        "subject_locator_test.go",
    ],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/registry/rbac/validation:go_default_library",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
    ],
)

filegroup(
    name = "package-srcs",
    srcs = glob(["**"]),
    tags = ["automanaged"],
    visibility = ["//visibility:private"],
)

filegroup(
    name = "all-srcs",
    srcs = [
        ":package-srcs",
        "//plugin/pkg/auth/authorizer/rbac/bootstrappolicy:all-srcs",
    ],
    tags = ["automanaged"],
)
