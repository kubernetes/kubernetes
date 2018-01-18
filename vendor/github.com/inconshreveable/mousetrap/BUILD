load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = select({
        "@io_bazel_rules_go//go/platform:android": [
            "trap_others.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "trap_others.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "trap_others.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "trap_others.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "trap_others.go",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "trap_others.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "trap_others.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "trap_others.go",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "trap_others.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "trap_others.go",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "trap_windows.go",
            "trap_windows_1.4.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "github.com/inconshreveable/mousetrap",
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
