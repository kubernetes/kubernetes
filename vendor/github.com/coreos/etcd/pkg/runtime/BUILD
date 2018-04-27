load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = select({
        "@io_bazel_rules_go//go/platform:android": [
            "fds_other.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "fds_other.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "fds_other.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "fds_other.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "fds_linux.go",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "fds_other.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "fds_other.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "fds_other.go",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "fds_other.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "fds_other.go",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "fds_other.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "github.com/coreos/etcd/pkg/runtime",
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
