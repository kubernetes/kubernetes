load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "common.go",
        "config.go",
        "container.go",
        "network.go",
        "node.go",
        "runtime.go",
        "secret.go",
        "service.go",
        "swarm.go",
        "task.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/docker/docker/api/types/swarm",
    importpath = "github.com/docker/docker/api/types/swarm",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/docker/docker/api/types/container:go_default_library",
        "//vendor/github.com/docker/docker/api/types/mount:go_default_library",
        "//vendor/github.com/docker/docker/api/types/network:go_default_library",
        "//vendor/github.com/docker/docker/api/types/swarm/runtime:go_default_library",
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
        "//vendor/github.com/docker/docker/api/types/swarm/runtime:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
