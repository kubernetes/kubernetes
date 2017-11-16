package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "defaultfs.go",
        "fakefs.go",
        "filesystem.go",
        "watcher.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/util/filesystem",
    deps = [
        "//vendor/github.com/fsnotify/fsnotify:go_default_library",
        "//vendor/github.com/spf13/afero:go_default_library",
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
)
