load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = select({
        "@io_bazel_rules_go//go/platform:android": [
            "fsnotify.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "fsnotify.go",
            "kqueue.go",
            "open_mode_darwin.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "fsnotify.go",
            "kqueue.go",
            "open_mode_bsd.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "fsnotify.go",
            "kqueue.go",
            "open_mode_bsd.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "fsnotify.go",
            "inotify.go",
            "inotify_poller.go",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "fsnotify.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "fsnotify.go",
            "kqueue.go",
            "open_mode_bsd.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "fsnotify.go",
            "kqueue.go",
            "open_mode_bsd.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "fen.go",
            "fsnotify.go",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "fsnotify.go",
            "windows.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "github.com/fsnotify/fsnotify",
    visibility = ["//visibility:public"],
    deps = select({
        "@io_bazel_rules_go//go/platform:darwin": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
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
