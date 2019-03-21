load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "args.go",
        "bash_completions.go",
        "cobra.go",
        "command.go",
        "command_notwin.go",
        "command_win.go",
        "zsh_completions.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/spf13/cobra",
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
