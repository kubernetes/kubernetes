load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "popcnt_amd64.go",
        "popcnt_amd64.s",
        "popcnt_generic.go",
        "sparse.go",
        "util.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/golang.org/x/tools/container/intsets",
    importpath = "golang.org/x/tools/container/intsets",
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
