load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "markdown.go",
        "normalizer.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/util/normalizer",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/MakeNowJust/heredoc:go_default_library",
        "//vendor/github.com/russross/blackfriday:go_default_library",
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
