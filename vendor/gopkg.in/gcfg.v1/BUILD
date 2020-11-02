load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "errors.go",
        "go1_0.go",
        "go1_2.go",
        "read.go",
        "set.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/gopkg.in/gcfg.v1",
    importpath = "gopkg.in/gcfg.v1",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/gopkg.in/gcfg.v1/scanner:go_default_library",
        "//vendor/gopkg.in/gcfg.v1/token:go_default_library",
        "//vendor/gopkg.in/gcfg.v1/types:go_default_library",
        "//vendor/gopkg.in/warnings.v0:go_default_library",
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
    srcs = [
        ":package-srcs",
        "//vendor/gopkg.in/gcfg.v1/scanner:all-srcs",
        "//vendor/gopkg.in/gcfg.v1/token:all-srcs",
        "//vendor/gopkg.in/gcfg.v1/types:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
