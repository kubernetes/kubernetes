load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "lookup.go",
        "lookup_unsupported.go",
        "user.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:darwin_amd64": [
            "lookup_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:linux_amd64": [
            "lookup_unix.go",
        ],
        "//conditions:default": [],
    }),
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
