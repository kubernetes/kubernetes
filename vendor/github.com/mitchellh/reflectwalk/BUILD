load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "location.go",
        "location_string.go",
        "reflectwalk.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/mitchellh/reflectwalk",
    importpath = "github.com/mitchellh/reflectwalk",
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
