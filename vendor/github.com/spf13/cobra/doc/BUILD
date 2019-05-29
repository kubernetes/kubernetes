load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "man_docs.go",
        "md_docs.go",
        "rest_docs.go",
        "util.go",
        "yaml_docs.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/spf13/cobra/doc",
    importpath = "github.com/spf13/cobra/doc",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/cpuguy83/go-md2man/md2man:go_default_library",
        "//vendor/github.com/spf13/cobra:go_default_library",
        "//vendor/github.com/spf13/pflag:go_default_library",
        "//vendor/gopkg.in/yaml.v2:go_default_library",
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
