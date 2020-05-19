load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "helpers.go",
        "merge.go",
        "parser.go",
        "remove.go",
        "tofieldset.go",
        "typed.go",
        "union.go",
        "validate.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/sigs.k8s.io/structured-merge-diff/v3/typed",
    importpath = "sigs.k8s.io/structured-merge-diff/v3/typed",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/gopkg.in/yaml.v2:go_default_library",
        "//vendor/sigs.k8s.io/structured-merge-diff/v3/fieldpath:go_default_library",
        "//vendor/sigs.k8s.io/structured-merge-diff/v3/schema:go_default_library",
        "//vendor/sigs.k8s.io/structured-merge-diff/v3/value:go_default_library",
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
