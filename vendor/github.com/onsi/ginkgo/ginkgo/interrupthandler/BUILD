load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "interrupt_handler.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:darwin": [
            "sigquit_swallower_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "sigquit_swallower_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "sigquit_swallower_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "sigquit_swallower_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "sigquit_swallower_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "sigquit_swallower_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "sigquit_swallower_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "sigquit_swallower_windows.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "github.com/onsi/ginkgo/ginkgo/interrupthandler",
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
