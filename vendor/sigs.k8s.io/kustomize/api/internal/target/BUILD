load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "kusttarget.go",
        "kusttarget_configplugin.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/sigs.k8s.io/kustomize/api/internal/target",
    importpath = "sigs.k8s.io/kustomize/api/internal/target",
    visibility = ["//vendor/sigs.k8s.io/kustomize/api:__subpackages__"],
    deps = [
        "//vendor/github.com/pkg/errors:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/api/builtins:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/api/ifc:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/api/internal/accumulator:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/api/internal/plugins/builtinconfig:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/api/internal/plugins/builtinhelpers:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/api/internal/plugins/loader:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/api/konfig:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/api/resmap:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/api/transform:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/api/types:go_default_library",
        "//vendor/sigs.k8s.io/yaml:go_default_library",
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
