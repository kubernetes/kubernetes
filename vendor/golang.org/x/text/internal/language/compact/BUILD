load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "compact.go",
        "language.go",
        "parents.go",
        "tables.go",
        "tags.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/golang.org/x/text/internal/language/compact",
    importpath = "golang.org/x/text/internal/language/compact",
    visibility = ["//vendor/golang.org/x/text:__subpackages__"],
    deps = ["//vendor/golang.org/x/text/internal/language:go_default_library"],
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
