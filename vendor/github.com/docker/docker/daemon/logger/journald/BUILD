load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "journald.go",
        "journald_unsupported.go",
        "read_unsupported.go",
    ],
    cgo = True,
    importmap = "k8s.io/kubernetes/vendor/github.com/docker/docker/daemon/logger/journald",
    importpath = "github.com/docker/docker/daemon/logger/journald",
    visibility = ["//visibility:public"],
    deps = select({
        "@io_bazel_rules_go//go/platform:linux": [
            "//vendor/github.com/coreos/go-systemd/journal:go_default_library",
            "//vendor/github.com/docker/docker/daemon/logger:go_default_library",
            "//vendor/github.com/docker/docker/daemon/logger/loggerutils:go_default_library",
            "//vendor/github.com/sirupsen/logrus:go_default_library",
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
