load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["build.go"],
    importmap = "k8s.io/kubernetes/vendor/sigs.k8s.io/kustomize/pkg/commands/build",
    importpath = "sigs.k8s.io/kustomize/pkg/commands/build",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/pkg/errors:go_default_library",
        "//vendor/github.com/spf13/cobra:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pkg/constants:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pkg/fs:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pkg/ifc/transformer:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pkg/loader:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pkg/resmap:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pkg/target:go_default_library",
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
