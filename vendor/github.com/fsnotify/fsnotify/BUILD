load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "fsnotify.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:darwin_amd64": [
            "kqueue.go",
            "open_mode_darwin.go",
        ],
        "@io_bazel_rules_go//go/platform:linux_amd64": [
            "inotify.go",
            "inotify_poller.go",
        ],
        "@io_bazel_rules_go//go/platform:windows_amd64": [
            "windows.go",
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    deps = select({
        "@io_bazel_rules_go//go/platform:darwin_amd64": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:linux_amd64": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
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
