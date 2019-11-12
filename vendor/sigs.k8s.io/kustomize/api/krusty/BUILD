load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "kustomizer.go",
        "options.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/sigs.k8s.io/kustomize/api/krusty",
    importpath = "sigs.k8s.io/kustomize/api/krusty",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/sigs.k8s.io/kustomize/api/builtins:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/api/filesys:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/api/internal/k8sdeps/transformer:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/api/internal/plugins/loader:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/api/internal/target:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/api/k8sdeps/kunstruct:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/api/k8sdeps/validator:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/api/konfig:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/api/loader:go_default_library",
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
