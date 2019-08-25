load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = ["fix.go"],
    importmap = "k8s.io/kubernetes/vendor/sigs.k8s.io/kustomize/pkg/commands/edit/fix",
    importpath = "sigs.k8s.io/kustomize/pkg/commands/edit/fix",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/spf13/cobra:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pkg/commands/kustfile:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pkg/fs:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["fix_test.go"],
    embed = [":go_default_library"],
    tags = ["manual"],
    deps = ["//vendor/sigs.k8s.io/kustomize/pkg/fs:go_default_library"],
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
