load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["kv.go"],
    importmap = "k8s.io/kubernetes/vendor/github.com/docker/docker/pkg/discovery/kv",
    importpath = "github.com/docker/docker/pkg/discovery/kv",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/docker/docker/pkg/discovery:go_default_library",
        "//vendor/github.com/docker/go-connections/tlsconfig:go_default_library",
        "//vendor/github.com/docker/libkv:go_default_library",
        "//vendor/github.com/docker/libkv/store:go_default_library",
        "//vendor/github.com/docker/libkv/store/consul:go_default_library",
        "//vendor/github.com/docker/libkv/store/etcd:go_default_library",
        "//vendor/github.com/docker/libkv/store/zookeeper:go_default_library",
        "//vendor/github.com/sirupsen/logrus:go_default_library",
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
