load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "elements.go",
        "equals.go",
        "schemaschema.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/sigs.k8s.io/structured-merge-diff/schema",
    importpath = "sigs.k8s.io/structured-merge-diff/schema",
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

go_test(
    name = "go_default_test",
    srcs = [
        "elements_test.go",
        "equals_test.go",
    ],
    embed = [":go_default_library"],
    tags = ["manual"],
    deps = ["//vendor/github.com/google/gofuzz:go_default_library"],
)
