load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "config.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "seccomp_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "seccomp_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "seccomp_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "seccomp_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "seccomp_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "seccomp_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "seccomp_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "seccomp_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "seccomp_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "seccomp_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "seccomp_unsupported.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "github.com/opencontainers/runc/libcontainer/seccomp",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/opencontainers/runc/libcontainer/configs:go_default_library",
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
