load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = select({
        "@io_bazel_rules_go//go/platform:android": [
            "apparmor_disabled.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "apparmor_disabled.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "apparmor_disabled.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "apparmor_disabled.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "apparmor_disabled.go",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "apparmor_disabled.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "apparmor_disabled.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "apparmor_disabled.go",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "apparmor_disabled.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "apparmor_disabled.go",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "apparmor_disabled.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "github.com/opencontainers/runc/libcontainer/apparmor",
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
