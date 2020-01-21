load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "elements.go",
        "equals.go",
        "schemaschema.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/sigs.k8s.io/structured-merge-diff/v3/schema",
    importpath = "sigs.k8s.io/structured-merge-diff/v3/schema",
    visibility = ["//visibility:public"],
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
