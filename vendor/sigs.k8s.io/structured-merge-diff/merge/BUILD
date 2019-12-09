load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "conflict.go",
        "update.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/sigs.k8s.io/structured-merge-diff/merge",
    importpath = "sigs.k8s.io/structured-merge-diff/merge",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/sigs.k8s.io/structured-merge-diff/fieldpath:go_default_library",
        "//vendor/sigs.k8s.io/structured-merge-diff/typed:go_default_library",
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

go_test(
    name = "go_default_test",
    srcs = [
        "conflict_test.go",
        "deduced_test.go",
        "key_test.go",
        "leaf_test.go",
        "multiple_appliers_test.go",
        "nested_test.go",
        "obsolete_versions_test.go",
        "preserve_unknown_test.go",
        "real_test.go",
        "set_test.go",
        "union_test.go",
    ],
    embed = [":go_default_library"],
    tags = ["manual"],
    deps = [
        "//vendor/sigs.k8s.io/structured-merge-diff/fieldpath:go_default_library",
        "//vendor/sigs.k8s.io/structured-merge-diff/internal/fixture:go_default_library",
        "//vendor/sigs.k8s.io/structured-merge-diff/typed:go_default_library",
        "//vendor/sigs.k8s.io/structured-merge-diff/value:go_default_library",
    ],
)
