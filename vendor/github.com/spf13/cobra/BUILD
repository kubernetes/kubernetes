load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "args.go",
        "bash_completions.go",
        "cobra.go",
        "command.go",
        "zsh_completions.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "command_notwin.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "command_notwin.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "command_notwin.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "command_notwin.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "command_notwin.go",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "command_notwin.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "command_notwin.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "command_notwin.go",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "command_notwin.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "command_notwin.go",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "command_win.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "github.com/spf13/cobra",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/spf13/pflag:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:windows": [
            "//vendor/github.com/inconshreveable/mousetrap:go_default_library",
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
    srcs = [
        ":package-srcs",
        "//vendor/github.com/spf13/cobra/doc:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
