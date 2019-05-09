load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "ascii.go",
        "case.go",
        "legal.go",
        "mime.go",
        "notwords.go",
        "replace.go",
        "stringreplacer.go",
        "url.go",
        "words.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/client9/misspell",
    importpath = "github.com/client9/misspell",
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
    srcs = [
        ":package-srcs",
        "//vendor/github.com/client9/misspell/cmd/misspell:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
