load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "kusttarget.go",
        "kusttarget_configplugin.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/sigs.k8s.io/kustomize/api/target",
    importpath = "sigs.k8s.io/kustomize/api/target",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/pkg/errors:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/api/ifc:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/api/internal/accumulator:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/api/pgmconfig:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/api/plugins/builtinconfig:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/api/plugins/builtinhelpers:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/api/plugins/builtins:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/api/plugins/loader:go_default_library",
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
