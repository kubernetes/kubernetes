package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = ["storage_extensions.go"],
    importpath = "k8s.io/kubernetes/pkg/registry/extensions/rest",
    deps = [
        "//pkg/api/legacyscheme:go_default_library",
        "//pkg/apis/extensions:go_default_library",
        "//pkg/registry/extensions/controller/storage:go_default_library",
        "//pkg/registry/extensions/daemonset/storage:go_default_library",
        "//pkg/registry/extensions/deployment/storage:go_default_library",
        "//pkg/registry/extensions/ingress/storage:go_default_library",
        "//pkg/registry/extensions/podsecuritypolicy/storage:go_default_library",
        "//pkg/registry/extensions/replicaset/storage:go_default_library",
        "//pkg/registry/networking/networkpolicy/storage:go_default_library",
        "//vendor/k8s.io/api/extensions/v1beta1:go_default_library",
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
