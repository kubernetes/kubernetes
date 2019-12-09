load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = ["state.go"],
    importmap = "k8s.io/kubernetes/vendor/sigs.k8s.io/structured-merge-diff/internal/fixture",
    importpath = "sigs.k8s.io/structured-merge-diff/internal/fixture",
    visibility = ["//vendor/sigs.k8s.io/structured-merge-diff:__subpackages__"],
    deps = [
        "//vendor/sigs.k8s.io/structured-merge-diff/fieldpath:go_default_library",
        "//vendor/sigs.k8s.io/structured-merge-diff/merge:go_default_library",
        "//vendor/sigs.k8s.io/structured-merge-diff/typed:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["state_test.go"],
    embed = [":go_default_library"],
    tags = ["manual"],
    deps = ["//vendor/sigs.k8s.io/structured-merge-diff/typed:go_default_library"],
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
