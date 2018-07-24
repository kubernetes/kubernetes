load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["utf8internal.go"],
    importmap = "k8s.io/kubernetes/vendor/golang.org/x/text/internal/utf8internal",
    importpath = "golang.org/x/text/internal/utf8internal",
    visibility = ["//vendor/golang.org/x/text:__subpackages__"],
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
