load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "signal.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "signal_posix.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "signal_posix.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "signal_posix.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "signal_posix.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "signal_posix.go",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "signal_posix.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "signal_posix.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "signal_posix.go",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "signal_posix.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "signal_posix.go",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "signal_windows.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "k8s.io/sample-controller/pkg/signals",
    visibility = ["//visibility:public"],
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
