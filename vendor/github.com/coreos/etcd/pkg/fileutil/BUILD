load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "dir_unix.go",
        "fileutil.go",
        "lock.go",
        "lock_flock.go",
        "lock_unix.go",
        "preallocate.go",
        "preallocate_unsupported.go",
        "purge.go",
        "sync.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:darwin_amd64": [
            "preallocate_darwin.go",
            "sync_darwin.go",
        ],
        "@io_bazel_rules_go//go/platform:linux_amd64": [
            "lock_linux.go",
            "preallocate_unix.go",
            "sync_linux.go",
        ],
        "@io_bazel_rules_go//go/platform:windows_amd64": [
            "dir_windows.go",
            "lock_windows.go",
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    deps = ["//vendor/github.com/coreos/pkg/capnslog:go_default_library"],
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
