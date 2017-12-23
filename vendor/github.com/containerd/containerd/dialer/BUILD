load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "dialer.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "dialer_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "dialer_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "dialer_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "dialer_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "dialer_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "dialer_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "dialer_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "dialer_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "dialer_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "dialer_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "dialer_windows.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "github.com/containerd/containerd/dialer",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/pkg/errors:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:windows": [
            "//vendor/github.com/Microsoft/go-winio:go_default_library",
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
