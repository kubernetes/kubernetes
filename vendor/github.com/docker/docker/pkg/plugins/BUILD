load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "client.go",
        "discovery.go",
        "discovery_unix.go",
        "discovery_windows.go",
        "errors.go",
        "plugins.go",
        "plugins_unix.go",
        "plugins_windows.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/docker/docker/pkg/plugins",
    importpath = "github.com/docker/docker/pkg/plugins",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/docker/docker/pkg/plugins/transport:go_default_library",
        "//vendor/github.com/docker/go-connections/sockets:go_default_library",
        "//vendor/github.com/docker/go-connections/tlsconfig:go_default_library",
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
    srcs = [
        ":package-srcs",
        "//vendor/github.com/docker/docker/pkg/plugins/pluginrpc-gen:all-srcs",
        "//vendor/github.com/docker/docker/pkg/plugins/transport:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
