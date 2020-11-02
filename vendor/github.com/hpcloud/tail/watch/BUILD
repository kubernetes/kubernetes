load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "filechanges.go",
        "inotify.go",
        "inotify_tracker.go",
        "polling.go",
        "watch.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/hpcloud/tail/watch",
    importpath = "github.com/hpcloud/tail/watch",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/hpcloud/tail/util:go_default_library",
        "//vendor/gopkg.in/fsnotify.v1:go_default_library",
        "//vendor/gopkg.in/tomb.v1:go_default_library",
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
