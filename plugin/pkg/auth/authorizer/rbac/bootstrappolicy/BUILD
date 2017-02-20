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
        "controller_policy.go",
        "policy.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:k8s.io/apimachinery/pkg/apis/meta/v1"],
)

go_test(
    name = "go_default_xtest",
    srcs = ["policy_test.go"],
    data = glob([
        "testdata/*",
    ]),
    tags = ["automanaged"],
    deps = [
        "//pkg/registry/rbac/validation:go_default_library",
        "//plugin/pkg/auth/authorizer/rbac/bootstrappolicy:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["controller_policy_test.go"],
    library = ":go_default_library",
    tags = ["automanaged"],
)

filegroup(
    name = "package-srcs",
    srcs = glob(["**"]),
    tags = ["automanaged"],
    visibility = ["//visibility:private"],
)

filegroup(
    name = "all-srcs",
    srcs = [":package-srcs"],
    tags = ["automanaged"],
)
