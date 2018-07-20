load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "config.go",
        "container.go",
        "network.go",
        "node.go",
        "secret.go",
        "service.go",
        "swarm.go",
        "task.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/docker/docker/daemon/cluster/convert",
    importpath = "github.com/docker/docker/daemon/cluster/convert",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/docker/docker/api/types:go_default_library",
        "//vendor/github.com/docker/docker/api/types/container:go_default_library",
        "//vendor/github.com/docker/docker/api/types/mount:go_default_library",
        "//vendor/github.com/docker/docker/api/types/network:go_default_library",
        "//vendor/github.com/docker/docker/api/types/swarm:go_default_library",
        "//vendor/github.com/docker/docker/api/types/swarm/runtime:go_default_library",
        "//vendor/github.com/docker/docker/pkg/namesgenerator:go_default_library",
        "//vendor/github.com/docker/libnetwork/datastore:go_default_library",
        "//vendor/github.com/docker/swarmkit/api:go_default_library",
        "//vendor/github.com/docker/swarmkit/api/genericresource:go_default_library",
        "//vendor/github.com/docker/swarmkit/ca:go_default_library",
        "//vendor/github.com/gogo/protobuf/proto:go_default_library",
        "//vendor/github.com/gogo/protobuf/types:go_default_library",
        "//vendor/github.com/pkg/errors:go_default_library",
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
