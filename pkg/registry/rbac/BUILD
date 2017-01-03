package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = ["escalation_check.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/genericapiserver/api/request:go_default_library",
        "//vendor:k8s.io/apiserver/pkg/authentication/user",
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
        "//pkg/registry/rbac/clusterrole:all-srcs",
        "//pkg/registry/rbac/clusterrolebinding:all-srcs",
        "//pkg/registry/rbac/rest:all-srcs",
        "//pkg/registry/rbac/role:all-srcs",
        "//pkg/registry/rbac/rolebinding:all-srcs",
    ],
    tags = ["automanaged"],
)
