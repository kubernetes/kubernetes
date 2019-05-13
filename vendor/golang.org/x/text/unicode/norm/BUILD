load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "composition.go",
        "forminfo.go",
        "input.go",
        "iter.go",
        "normalize.go",
        "readwriter.go",
        "tables10.0.0.go",
        "tables9.0.0.go",
        "transform.go",
        "trie.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/golang.org/x/text/unicode/norm",
    importpath = "golang.org/x/text/unicode/norm",
    visibility = ["//visibility:public"],
    deps = ["//vendor/golang.org/x/text/transform:go_default_library"],
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
