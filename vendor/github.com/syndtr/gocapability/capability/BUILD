load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "capability.go",
        "enum.go",
        "enum_gen.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "capability_noop.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "capability_noop.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "capability_noop.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "capability_noop.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "capability_linux.go",
            "syscall_linux.go",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "capability_noop.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "capability_noop.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "capability_noop.go",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "capability_noop.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "capability_noop.go",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "capability_noop.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "github.com/syndtr/gocapability/capability",
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
