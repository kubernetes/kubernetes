load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "command_groups.go",
        "markdown.go",
        "normalizers.go",
        "templater.go",
        "templates.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/kubectl/pkg/util/templates",
    importpath = "k8s.io/kubectl/pkg/util/templates",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/kubectl/pkg/util/term:go_default_library",
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
    visibility = ["//visibility:private"],
)

filegroup(
    name = "all-srcs",
    srcs = [":package-srcs"],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)

go_test(
    name = "go_default_test",
    srcs = ["normalizers_test.go"],
    embed = [":go_default_library"],
)
