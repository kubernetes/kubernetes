load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "experimental.go",
        "local.go",
        "router.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/docker/docker/api/server/router",
    importpath = "github.com/docker/docker/api/server/router",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/docker/docker/api/server/httputils:go_default_library",
        "//vendor/golang.org/x/net/context:go_default_library",
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
        "//vendor/github.com/docker/docker/api/server/router/checkpoint:all-srcs",
        "//vendor/github.com/docker/docker/api/server/router/container:all-srcs",
        "//vendor/github.com/docker/docker/api/server/router/debug:all-srcs",
        "//vendor/github.com/docker/docker/api/server/router/distribution:all-srcs",
        "//vendor/github.com/docker/docker/api/server/router/image:all-srcs",
        "//vendor/github.com/docker/docker/api/server/router/network:all-srcs",
        "//vendor/github.com/docker/docker/api/server/router/plugin:all-srcs",
        "//vendor/github.com/docker/docker/api/server/router/session:all-srcs",
        "//vendor/github.com/docker/docker/api/server/router/swarm:all-srcs",
        "//vendor/github.com/docker/docker/api/server/router/system:all-srcs",
        "//vendor/github.com/docker/docker/api/server/router/volume:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
