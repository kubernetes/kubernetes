package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
    "go_test",
    "cgo_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "command_groups.go",
        "markdown.go",
        "normalizers.go",
        "templater.go",
        "templates.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/util/term:go_default_library",
        "//vendor:github.com/MakeNowJust/heredoc",
        "//vendor:github.com/russross/blackfriday",
        "//vendor:github.com/spf13/cobra",
        "//vendor:github.com/spf13/pflag",
    ],
)
