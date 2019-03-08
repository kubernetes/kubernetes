load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "fsnotify.go",
        "inotify.go",
        "inotify_poller.go",
        "kqueue.go",
        "open_mode_bsd.go",
        "open_mode_darwin.go",
        "windows.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/gopkg.in/fsnotify/fsnotify.v1",
    importpath = "gopkg.in/fsnotify/fsnotify.v1",
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
