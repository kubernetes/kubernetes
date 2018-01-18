package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = ["plugins.go"],
    importpath = "k8s.io/client-go/plugin/pkg/client/auth",
    deps = [
        "//vendor/k8s.io/client-go/plugin/pkg/client/auth/azure:go_default_library",
        "//vendor/k8s.io/client-go/plugin/pkg/client/auth/gcp:go_default_library",
        "//vendor/k8s.io/client-go/plugin/pkg/client/auth/oidc:go_default_library",
        "//vendor/k8s.io/client-go/plugin/pkg/client/auth/openstack:go_default_library",
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
        "//staging/src/k8s.io/client-go/plugin/pkg/client/auth/azure:all-srcs",
        "//staging/src/k8s.io/client-go/plugin/pkg/client/auth/gcp:all-srcs",
        "//staging/src/k8s.io/client-go/plugin/pkg/client/auth/oidc:all-srcs",
        "//staging/src/k8s.io/client-go/plugin/pkg/client/auth/openstack:all-srcs",
    ],
    tags = ["automanaged"],
)
