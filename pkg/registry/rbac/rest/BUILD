package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = ["storage_rbac.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/client/clientset_generated/internalclientset/typed/rbac/internalversion:go_default_library",
        "//pkg/client/retry:go_default_library",
        "//pkg/registry/rbac/clusterrole:go_default_library",
        "//pkg/registry/rbac/clusterrole/policybased:go_default_library",
        "//pkg/registry/rbac/clusterrole/storage:go_default_library",
        "//pkg/registry/rbac/clusterrolebinding:go_default_library",
        "//pkg/registry/rbac/clusterrolebinding/policybased:go_default_library",
        "//pkg/registry/rbac/clusterrolebinding/storage:go_default_library",
        "//pkg/registry/rbac/reconciliation:go_default_library",
        "//pkg/registry/rbac/role:go_default_library",
        "//pkg/registry/rbac/role/policybased:go_default_library",
        "//pkg/registry/rbac/role/storage:go_default_library",
        "//pkg/registry/rbac/rolebinding:go_default_library",
        "//pkg/registry/rbac/rolebinding/policybased:go_default_library",
        "//pkg/registry/rbac/rolebinding/storage:go_default_library",
        "//pkg/registry/rbac/validation:go_default_library",
        "//plugin/pkg/auth/authorizer/rbac/bootstrappolicy:go_default_library",
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
    srcs = [":package-srcs"],
    tags = ["automanaged"],
)
