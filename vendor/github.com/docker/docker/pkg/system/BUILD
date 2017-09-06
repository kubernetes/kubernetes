load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "chtimes.go",
        "chtimes_unix.go",
        "errors.go",
        "exitcode.go",
        "filesys.go",
        "init.go",
        "lcow_unix.go",
        "lstat_unix.go",
        "meminfo.go",
        "meminfo_unsupported.go",
        "mknod.go",
        "path.go",
        "path_unix.go",
        "rm.go",
        "stat_unix.go",
        "umask.go",
        "utimes_unsupported.go",
        "xattrs_unsupported.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:darwin_amd64": [
            "process_unix.go",
            "stat_darwin.go",
        ],
        "@io_bazel_rules_go//go/platform:linux_amd64": [
            "meminfo_linux.go",
            "process_unix.go",
            "stat_linux.go",
            "syscall_unix.go",
            "utimes_linux.go",
            "xattrs_linux.go",
        ],
        "@io_bazel_rules_go//go/platform:windows_amd64": [
            "chtimes_windows.go",
            "events_windows.go",
            "filesys_windows.go",
            "init_windows.go",
            "lcow_windows.go",
            "lstat_windows.go",
            "meminfo_windows.go",
            "mknod_windows.go",
            "path_windows.go",
            "stat_windows.go",
            "syscall_windows.go",
            "umask_windows.go",
        ],
        "//conditions:default": [],
    }),
    cgo = True,
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/docker/docker/pkg/mount:go_default_library",
        "//vendor/github.com/pkg/errors:go_default_library",
        "//vendor/golang.org/x/sys/unix:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:linux_amd64": [
            "//vendor/github.com/docker/go-units:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:windows_amd64": [
            "//vendor/github.com/Microsoft/go-winio:go_default_library",
            "//vendor/github.com/sirupsen/logrus:go_default_library",
            "//vendor/golang.org/x/sys/windows:go_default_library",
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
