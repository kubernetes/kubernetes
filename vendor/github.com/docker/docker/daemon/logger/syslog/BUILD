load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["syslog.go"],
    importmap = "k8s.io/kubernetes/vendor/github.com/docker/docker/daemon/logger/syslog",
    importpath = "github.com/docker/docker/daemon/logger/syslog",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/RackSec/srslog:go_default_library",
        "//vendor/github.com/docker/docker/daemon/logger:go_default_library",
        "//vendor/github.com/docker/docker/daemon/logger/loggerutils:go_default_library",
        "//vendor/github.com/docker/docker/pkg/urlutil:go_default_library",
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
    srcs = [":package-srcs"],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
