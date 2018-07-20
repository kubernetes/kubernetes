load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["logentries.go"],
    importmap = "k8s.io/kubernetes/vendor/github.com/docker/docker/daemon/logger/logentries",
    importpath = "github.com/docker/docker/daemon/logger/logentries",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/bsphere/le_go:go_default_library",
        "//vendor/github.com/docker/docker/daemon/logger:go_default_library",
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
