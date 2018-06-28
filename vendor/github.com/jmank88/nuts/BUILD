load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "key.go",
        "nuts.go",
        "paths.go",
        "types.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/jmank88/nuts",
    importpath = "github.com/jmank88/nuts",
    visibility = ["//visibility:public"],
    deps = ["//vendor/github.com/boltdb/bolt:go_default_library"],
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
