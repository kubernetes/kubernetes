package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = ["storage_settings.go"],
    importpath = "k8s.io/kubernetes/pkg/registry/networking/rest",
    deps = [
        "//pkg/api/legacyscheme:go_default_library",
        "//pkg/apis/networking:go_default_library",
        "//pkg/registry/networking/ingress/storage:go_default_library",
        "//pkg/registry/networking/ingressclass/storage:go_default_library",
        "//pkg/registry/networking/networkpolicy/storage:go_default_library",
        "//staging/src/k8s.io/api/networking/v1:go_default_library",
        "//staging/src/k8s.io/api/networking/v1beta1:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/registry/generic:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/registry/rest:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/server:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/server/storage:go_default_library",
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
