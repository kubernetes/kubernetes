load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "userns.go",
        "userns_linux.go",
        "userns_unsupported.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/opencontainers/runc/libcontainer/userns",
    importpath = "github.com/opencontainers/runc/libcontainer/userns",
    visibility = ["//visibility:public"],
    deps = select({
        "@io_bazel_rules_go//go/platform:aix": [
            "//vendor/github.com/opencontainers/runc/libcontainer/user:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:android": [
            "//vendor/github.com/opencontainers/runc/libcontainer/user:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "//vendor/github.com/opencontainers/runc/libcontainer/user:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "//vendor/github.com/opencontainers/runc/libcontainer/user:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "//vendor/github.com/opencontainers/runc/libcontainer/user:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:illumos": [
            "//vendor/github.com/opencontainers/runc/libcontainer/user:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:ios": [
            "//vendor/github.com/opencontainers/runc/libcontainer/user:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:js": [
            "//vendor/github.com/opencontainers/runc/libcontainer/user:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "//vendor/github.com/opencontainers/runc/libcontainer/user:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "//vendor/github.com/opencontainers/runc/libcontainer/user:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "//vendor/github.com/opencontainers/runc/libcontainer/user:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "//vendor/github.com/opencontainers/runc/libcontainer/user:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "//vendor/github.com/opencontainers/runc/libcontainer/user:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "//vendor/github.com/opencontainers/runc/libcontainer/user:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "//vendor/github.com/opencontainers/runc/libcontainer/user:go_default_library",
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
