load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "fileutil.go",
        "lock.go",
        "preallocate.go",
        "purge.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "dir_unix.go",
            "lock_flock.go",
            "lock_unix.go",
            "preallocate_unsupported.go",
            "sync.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "dir_unix.go",
            "lock_flock.go",
            "lock_unix.go",
            "preallocate_darwin.go",
            "sync_darwin.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "dir_unix.go",
            "lock_flock.go",
            "lock_unix.go",
            "preallocate_unsupported.go",
            "sync.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "dir_unix.go",
            "lock_flock.go",
            "lock_unix.go",
            "preallocate_unsupported.go",
            "sync.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "dir_unix.go",
            "lock_flock.go",
            "lock_linux.go",
            "preallocate_unix.go",
            "sync_linux.go",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "dir_unix.go",
            "lock_flock.go",
            "lock_unix.go",
            "preallocate_unsupported.go",
            "sync.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "dir_unix.go",
            "lock_flock.go",
            "lock_unix.go",
            "preallocate_unsupported.go",
            "sync.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "dir_unix.go",
            "lock_flock.go",
            "lock_unix.go",
            "preallocate_unsupported.go",
            "sync.go",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "dir_unix.go",
            "lock_plan9.go",
            "preallocate_unsupported.go",
            "sync.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "dir_unix.go",
            "lock_solaris.go",
            "preallocate_unsupported.go",
            "sync.go",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "dir_windows.go",
            "lock_windows.go",
            "preallocate_unsupported.go",
            "sync.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "github.com/coreos/etcd/pkg/fileutil",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/coreos/pkg/capnslog:go_default_library",
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
