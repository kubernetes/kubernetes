load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "config.go",
        "config_common_unix.go",
        "config_unix.go",
        "config_windows.go",
        "opts.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/docker/docker/daemon/config",
    importpath = "github.com/docker/docker/daemon/config",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/docker/docker/api/types/swarm:go_default_library",
        "//vendor/github.com/docker/docker/daemon/cluster/convert:go_default_library",
        "//vendor/github.com/docker/docker/daemon/discovery:go_default_library",
        "//vendor/github.com/docker/docker/opts:go_default_library",
        "//vendor/github.com/docker/docker/pkg/authorization:go_default_library",
        "//vendor/github.com/docker/docker/pkg/discovery:go_default_library",
        "//vendor/github.com/docker/docker/registry:go_default_library",
        "//vendor/github.com/docker/swarmkit/api/genericresource:go_default_library",
        "//vendor/github.com/imdario/mergo:go_default_library",
        "//vendor/github.com/sirupsen/logrus:go_default_library",
        "//vendor/github.com/spf13/pflag:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:freebsd": [
            "//vendor/github.com/docker/docker/api/types:go_default_library",
            "//vendor/github.com/docker/docker/api/types/container:go_default_library",
            "//vendor/github.com/docker/go-units:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "//vendor/github.com/docker/docker/api/types:go_default_library",
            "//vendor/github.com/docker/docker/api/types/container:go_default_library",
            "//vendor/github.com/docker/go-units:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "//vendor/github.com/docker/docker/api/types:go_default_library",
        ],
        "//conditions:default": [],
    }),
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
