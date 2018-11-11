load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "api.go",
        "doc.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/pquerna/cachecontrol",
    importpath = "github.com/pquerna/cachecontrol",
    visibility = ["//visibility:public"],
    deps = ["//vendor/github.com/pquerna/cachecontrol/cacheobject:go_default_library"],
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
        "//vendor/github.com/pquerna/cachecontrol/cacheobject:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
