load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "errors.go",
        "scanner.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/gopkg.in/gcfg.v1/scanner",
    importpath = "gopkg.in/gcfg.v1/scanner",
    visibility = ["//visibility:public"],
    deps = ["//vendor/gopkg.in/gcfg.v1/token:go_default_library"],
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
