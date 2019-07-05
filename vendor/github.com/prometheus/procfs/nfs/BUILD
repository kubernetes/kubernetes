load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "nfs.go",
        "parse.go",
        "parse_nfs.go",
        "parse_nfsd.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/prometheus/procfs/nfs",
    importpath = "github.com/prometheus/procfs/nfs",
    visibility = ["//visibility:public"],
    deps = ["//vendor/github.com/prometheus/procfs/internal/util:go_default_library"],
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
