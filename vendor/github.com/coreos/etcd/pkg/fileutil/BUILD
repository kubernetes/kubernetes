load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "dir_unix.go",
        "dir_windows.go",
        "fileutil.go",
        "lock.go",
        "lock_flock.go",
        "lock_linux.go",
        "lock_plan9.go",
        "lock_solaris.go",
        "lock_unix.go",
        "lock_windows.go",
        "preallocate.go",
        "preallocate_darwin.go",
        "preallocate_unix.go",
        "preallocate_unsupported.go",
        "purge.go",
        "sync.go",
        "sync_darwin.go",
        "sync_linux.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/coreos/etcd/pkg/fileutil",
    importpath = "github.com/coreos/etcd/pkg/fileutil",
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
