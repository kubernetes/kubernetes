load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "console.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:darwin": [
            "console_unix.go",
            "tc_darwin.go",
            "tc_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "console_unix.go",
            "tc_freebsd.go",
            "tc_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "console_linux.go",
            "console_unix.go",
            "tc_linux.go",
            "tc_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "console_unix.go",
            "tc_solaris_cgo.go",
            "tc_solaris_nocgo.go",
            "tc_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "console_windows.go",
        ],
        "//conditions:default": [],
    }),
    cgo = True,
    importpath = "github.com/containerd/console",
    visibility = ["//visibility:public"],
    deps = select({
        "@io_bazel_rules_go//go/platform:darwin": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "//vendor/github.com/pkg/errors:go_default_library",
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
