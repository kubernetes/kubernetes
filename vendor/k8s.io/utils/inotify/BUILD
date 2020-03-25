load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "inotify.go",
        "inotify_linux.go",
        "inotify_others.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/utils/inotify",
    importpath = "k8s.io/utils/inotify",
    visibility = ["//visibility:public"],
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
