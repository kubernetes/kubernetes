load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "netns.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:darwin_amd64": [
            "netns_unspecified.go",
        ],
        "@io_bazel_rules_go//go/platform:linux_amd64": [
            "netns_linux.go",
        ],
        "@io_bazel_rules_go//go/platform:windows_amd64": [
            "netns_unspecified.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "github.com/vishvananda/netns",
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
