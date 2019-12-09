load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "element.go",
        "fromvalue.go",
        "managers.go",
        "path.go",
        "pathelementmap.go",
        "serialize.go",
        "serialize-pe.go",
        "set.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/sigs.k8s.io/structured-merge-diff/fieldpath",
    importpath = "sigs.k8s.io/structured-merge-diff/fieldpath",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/json-iterator/go:go_default_library",
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
        "element_test.go",
        "fromvalue_test.go",
        "managers_test.go",
        "path_test.go",
        "pathelementmap_test.go",
        "serialize-pe_test.go",
        "serialize_test.go",
        "set_test.go",
    ],
    embed = [":go_default_library"],
    tags = ["manual"],
    deps = ["//vendor/sigs.k8s.io/structured-merge-diff/value:go_default_library"],
)
