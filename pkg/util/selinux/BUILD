package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "selinux.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "selinux_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "selinux_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "selinux_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "selinux_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "selinux_linux.go",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "selinux_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "selinux_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "selinux_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "selinux_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "selinux_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "selinux_unsupported.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "k8s.io/kubernetes/pkg/util/selinux",
    deps = select({
        "@io_bazel_rules_go//go/platform:linux": [
            "//vendor/github.com/opencontainers/selinux/go-selinux:go_default_library",
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
)
