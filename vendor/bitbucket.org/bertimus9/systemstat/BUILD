load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "systemstat.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:linux_amd64": [
            "systemstat_linux.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "bitbucket.org/bertimus9/systemstat",
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
