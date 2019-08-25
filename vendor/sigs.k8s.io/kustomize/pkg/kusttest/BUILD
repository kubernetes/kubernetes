load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["kusttestharness.go"],
    importmap = "k8s.io/kubernetes/vendor/sigs.k8s.io/kustomize/pkg/kusttest",
    importpath = "sigs.k8s.io/kustomize/pkg/kusttest",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/sigs.k8s.io/kustomize/internal/loadertest:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/k8sdeps/kunstruct:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/k8sdeps/transformer:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pkg/loader:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pkg/pgmconfig:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pkg/plugins:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pkg/resmap:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pkg/resource:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pkg/target:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pkg/transformers/config/defaultconfig:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pkg/types:go_default_library",
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
