load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "coverage.go",
        "doc.go",
        "go1_1.go",
        "go1_2.go",
        "language.go",
        "match.go",
        "parse.go",
        "tables.go",
        "tags.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/golang.org/x/text/language",
    importpath = "golang.org/x/text/language",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/golang.org/x/text/internal/language:go_default_library",
        "//vendor/golang.org/x/text/internal/language/compact:go_default_library",
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
