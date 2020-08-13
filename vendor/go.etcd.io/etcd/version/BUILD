load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["version.go"],
    importmap = "k8s.io/kubernetes/vendor/go.etcd.io/etcd/version",
    importpath = "go.etcd.io/etcd/version",
    visibility = ["//visibility:public"],
    deps = ["//vendor/github.com/coreos/go-semver/semver:go_default_library"],
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
