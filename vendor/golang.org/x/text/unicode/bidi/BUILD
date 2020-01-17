load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "bidi.go",
        "bracket.go",
        "core.go",
        "prop.go",
        "tables10.0.0.go",
        "tables11.0.0.go",
        "tables9.0.0.go",
        "trieval.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/golang.org/x/text/unicode/bidi",
    importpath = "golang.org/x/text/unicode/bidi",
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
