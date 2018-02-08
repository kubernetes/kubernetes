load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = select({
        "@io_bazel_rules_go//go/platform:android": [
            "label.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "label.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "label.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "label.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "label.go",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "label.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "label.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "label.go",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "label.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "label.go",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "label.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "github.com/opencontainers/selinux/go-selinux/label",
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
