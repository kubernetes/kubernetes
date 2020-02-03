load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "rbuf.go",
        "snap.go",
        "unsnap.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/glycerine/go-unsnap-stream",
    importpath = "github.com/glycerine/go-unsnap-stream",
    visibility = ["//visibility:public"],
    deps = ["//vendor/github.com/golang/snappy:go_default_library"],
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
