load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["config.go"],
    importmap = "k8s.io/kubernetes/vendor/honnef.co/go/tools/config",
    importpath = "honnef.co/go/tools/config",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/BurntSushi/toml:go_default_library",
        "//vendor/golang.org/x/tools/go/analysis:go_default_library",
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
