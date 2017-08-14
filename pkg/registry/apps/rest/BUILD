package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = ["storage_apps.go"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/apis/apps:go_default_library",
        "//pkg/registry/apps/controllerrevision/storage:go_default_library",
        "//pkg/registry/apps/statefulset/storage:go_default_library",
        "//pkg/registry/extensions/daemonset/storage:go_default_library",
        "//pkg/registry/extensions/deployment/storage:go_default_library",
        "//pkg/registry/extensions/replicaset/storage:go_default_library",
        "//vendor/k8s.io/api/apps/v1beta1:go_default_library",
        "//vendor/k8s.io/api/apps/v1beta2:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/registry/generic:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/registry/rest:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/server:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/server/storage:go_default_library",
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
