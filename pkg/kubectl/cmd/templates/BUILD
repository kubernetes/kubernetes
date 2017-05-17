licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
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
    visibility = [
        "//build/visible_to:pkg_kubectl_cmd_templates_CONSUMERS",
    ],
    deps = [
        "//pkg/util/term:go_default_library",
        "//vendor/github.com/MakeNowJust/heredoc:go_default_library",
        "//vendor/github.com/russross/blackfriday:go_default_library",
        "//vendor/github.com/spf13/cobra:go_default_library",
        "//vendor/github.com/spf13/pflag:go_default_library",
    ],
)

filegroup(
    name = "package-srcs",
    srcs = glob(["**"]),
    tags = ["automanaged"],
)

filegroup(
    name = "all-srcs",
    srcs = [":package-srcs"],
    tags = ["automanaged"],
    visibility = [
        "//build/visible_to:pkg_kubectl_cmd_templates_CONSUMERS",
    ],
)
