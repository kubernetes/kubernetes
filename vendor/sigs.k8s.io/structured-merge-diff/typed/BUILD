load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "helpers.go",
        "merge.go",
        "parser.go",
        "remove.go",
        "typed.go",
        "union.go",
        "validate.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/sigs.k8s.io/structured-merge-diff/typed",
    importpath = "sigs.k8s.io/structured-merge-diff/typed",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/gopkg.in/yaml.v2:go_default_library",
        "//vendor/sigs.k8s.io/structured-merge-diff/fieldpath:go_default_library",
        "//vendor/sigs.k8s.io/structured-merge-diff/schema:go_default_library",
        "//vendor/sigs.k8s.io/structured-merge-diff/value:go_default_library",
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
        "deduced_test.go",
        "merge_test.go",
        "parser_test.go",
        "remove_test.go",
        "symdiff_test.go",
        "toset_test.go",
        "union_test.go",
        "validate_test.go",
    ],
    embed = [":go_default_library"],
    tags = ["manual"],
    deps = [
        "//vendor/gopkg.in/yaml.v2:go_default_library",
        "//vendor/sigs.k8s.io/structured-merge-diff/fieldpath:go_default_library",
        "//vendor/sigs.k8s.io/structured-merge-diff/schema:go_default_library",
        "//vendor/sigs.k8s.io/structured-merge-diff/value:go_default_library",
    ],
)
