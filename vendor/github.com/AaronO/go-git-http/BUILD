load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "errors.go",
        "events.go",
        "git_reader.go",
        "githttp.go",
        "pktparser.go",
        "routing.go",
        "rpc_reader.go",
        "utils.go",
        "version.go",
    ],
    importpath = "github.com/AaronO/go-git-http",
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
