load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "chtimes.go",
        "chtimes_unix.go",
        "chtimes_windows.go",
        "errors.go",
        "events_windows.go",
        "exitcode.go",
        "filesys.go",
        "filesys_windows.go",
        "init.go",
        "init_windows.go",
        "lcow_unix.go",
        "lcow_windows.go",
        "lstat_unix.go",
        "lstat_windows.go",
        "meminfo.go",
        "meminfo_linux.go",
        "meminfo_solaris.go",
        "meminfo_unsupported.go",
        "meminfo_windows.go",
        "mknod.go",
        "mknod_windows.go",
        "path.go",
        "path_unix.go",
        "path_windows.go",
        "process_unix.go",
        "rm.go",
        "stat_darwin.go",
        "stat_freebsd.go",
        "stat_linux.go",
        "stat_openbsd.go",
        "stat_solaris.go",
        "stat_unix.go",
        "stat_windows.go",
        "syscall_unix.go",
        "syscall_windows.go",
        "umask.go",
        "umask_windows.go",
        "utimes_freebsd.go",
        "utimes_linux.go",
        "utimes_unsupported.go",
        "xattrs_linux.go",
        "xattrs_unsupported.go",
    ],
    cgo = True,
    clinkopts = select({
        "@io_bazel_rules_go//go/platform:solaris": [
            "-lkstat",
        ],
        "//conditions:default": [],
    }),
    copts = select({
        "@io_bazel_rules_go//go/platform:solaris": [
            "-std=c99",
        ],
        "//conditions:default": [],
    }),
    importmap = "k8s.io/kubernetes/vendor/github.com/docker/docker/pkg/system",
    importpath = "github.com/docker/docker/pkg/system",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/docker/docker/pkg/mount:go_default_library",
        "//vendor/github.com/pkg/errors:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
        ],
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
            "//vendor/github.com/docker/go-units:go_default_library",
            "//vendor/golang.org/x/sys/unix:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
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
