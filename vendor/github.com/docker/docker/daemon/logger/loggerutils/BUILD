load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "log_tag.go",
        "logfile.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/docker/docker/daemon/logger/loggerutils",
    importpath = "github.com/docker/docker/daemon/logger/loggerutils",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/docker/docker/daemon/logger:go_default_library",
        "//vendor/github.com/docker/docker/daemon/logger/loggerutils/multireader:go_default_library",
        "//vendor/github.com/docker/docker/daemon/logger/templates:go_default_library",
        "//vendor/github.com/docker/docker/pkg/filenotify:go_default_library",
        "//vendor/github.com/docker/docker/pkg/pubsub:go_default_library",
        "//vendor/github.com/docker/docker/pkg/tailfile:go_default_library",
        "//vendor/github.com/fsnotify/fsnotify:go_default_library",
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
    srcs = [
        ":package-srcs",
        "//vendor/github.com/docker/docker/daemon/logger/loggerutils/multireader:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
