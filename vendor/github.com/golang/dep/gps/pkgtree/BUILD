load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "digest.go",
        "dirwalk.go",
        "ignored_ruleset.go",
        "pkgtree.go",
        "reachmap.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/golang/dep/gps/pkgtree",
    importpath = "github.com/golang/dep/gps/pkgtree",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/armon/go-radix:go_default_library",
        "//vendor/github.com/pkg/errors:go_default_library",
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
