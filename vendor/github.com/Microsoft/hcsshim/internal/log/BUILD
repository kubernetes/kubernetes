load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["g.go"],
    importmap = "k8s.io/kubernetes/vendor/github.com/Microsoft/hcsshim/internal/log",
    importpath = "github.com/Microsoft/hcsshim/internal/log",
    visibility = ["//vendor/github.com/Microsoft/hcsshim:__subpackages__"],
    deps = [
        "//vendor/github.com/sirupsen/logrus:go_default_library",
        "//vendor/go.opencensus.io/trace:go_default_library",
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
