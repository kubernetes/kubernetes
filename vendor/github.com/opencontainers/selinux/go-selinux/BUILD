load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = select({
        "@io_bazel_rules_go//go/platform:linux": [
            "selinux.go",
            "xattrs.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "github.com/opencontainers/selinux/go-selinux",
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
    srcs = [
        ":package-srcs",
        "//vendor/github.com/opencontainers/selinux/go-selinux/label:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
