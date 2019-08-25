load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["all.go"],
    importmap = "k8s.io/kubernetes/vendor/sigs.k8s.io/kustomize/pkg/commands/edit",
    importpath = "sigs.k8s.io/kustomize/pkg/commands/edit",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/spf13/cobra:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pkg/commands/edit/add:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pkg/commands/edit/fix:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pkg/commands/edit/remove:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pkg/commands/edit/set:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pkg/fs:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pkg/ifc:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pkg/loader:go_default_library",
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
        "//vendor/sigs.k8s.io/kustomize/pkg/commands/edit/add:all-srcs",
        "//vendor/sigs.k8s.io/kustomize/pkg/commands/edit/fix:all-srcs",
        "//vendor/sigs.k8s.io/kustomize/pkg/commands/edit/remove:all-srcs",
        "//vendor/sigs.k8s.io/kustomize/pkg/commands/edit/set:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
