load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["truncindex.go"],
    importmap = "k8s.io/kubernetes/vendor/github.com/docker/docker/pkg/truncindex",
    importpath = "github.com/docker/docker/pkg/truncindex",
    visibility = ["//visibility:public"],
    deps = ["//vendor/github.com/tchap/go-patricia/patricia:go_default_library"],
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
