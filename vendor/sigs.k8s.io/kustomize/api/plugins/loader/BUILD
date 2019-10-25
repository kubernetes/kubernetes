load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["loader.go"],
    importmap = "k8s.io/kubernetes/vendor/sigs.k8s.io/kustomize/api/plugins/loader",
    importpath = "sigs.k8s.io/kustomize/api/plugins/loader",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/pkg/errors:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/api/ifc:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/api/plugins/builtinhelpers:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/api/plugins/config:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/api/plugins/execplugin:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/api/resid:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/api/resmap:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/api/resource:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/api/types:go_default_library",
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
    visibility = ["//visibility:public"],
)
