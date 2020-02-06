load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["report.go"],
    importmap = "k8s.io/kubernetes/vendor/gotest.tools/gotestsum/internal/junitxml",
    importpath = "gotest.tools/gotestsum/internal/junitxml",
    visibility = ["//vendor/gotest.tools/gotestsum:__subpackages__"],
    deps = [
        "//vendor/github.com/pkg/errors:go_default_library",
        "//vendor/github.com/sirupsen/logrus:go_default_library",
        "//vendor/gotest.tools/gotestsum/testjson:go_default_library",
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
