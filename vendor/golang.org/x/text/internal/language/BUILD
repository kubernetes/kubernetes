load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "common.go",
        "compact.go",
        "compose.go",
        "coverage.go",
        "language.go",
        "lookup.go",
        "match.go",
        "parse.go",
        "tables.go",
        "tags.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/golang.org/x/text/internal/language",
    importpath = "golang.org/x/text/internal/language",
    visibility = ["//vendor/golang.org/x/text:__subpackages__"],
    deps = ["//vendor/golang.org/x/text/internal/tag:go_default_library"],
)

filegroup(
    name = "package-srcs",
    srcs = glob(["**"]),
    tags = ["automanaged"],
    visibility = ["//visibility:private"],
)

filegroup(
    name = "all-srcs",
    srcs = [
        ":package-srcs",
        "//vendor/golang.org/x/text/internal/language/compact:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
