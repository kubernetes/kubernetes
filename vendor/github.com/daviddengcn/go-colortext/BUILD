load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "ct.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "ct_ansi.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "ct_ansi.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "ct_ansi.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "ct_ansi.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "ct_ansi.go",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "ct_ansi.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "ct_ansi.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "ct_ansi.go",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "ct_ansi.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "ct_ansi.go",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "ct_win.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "github.com/daviddengcn/go-colortext",
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
