load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = ["policy_comparator.go"],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/component-helpers/auth/rbac/validation",
    importpath = "k8s.io/component-helpers/auth/rbac/validation",
    visibility = ["//visibility:public"],
    deps = ["//staging/src/k8s.io/api/rbac/v1:go_default_library"],
)

go_test(
    name = "go_default_test",
    srcs = ["policy_comparator_test.go"],
    embed = [":go_default_library"],
    deps = ["//staging/src/k8s.io/api/rbac/v1:go_default_library"],
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
    visibility = ["//visibility:public"],
)
