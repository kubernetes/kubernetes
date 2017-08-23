load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "popcnt_generic.go",
        "sparse.go",
        "util.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:darwin_amd64": [
            "popcnt_amd64.go",
            "popcnt_amd64.s",
        ],
        "@io_bazel_rules_go//go/platform:linux_amd64": [
            "popcnt_amd64.go",
            "popcnt_amd64.s",
        ],
        "@io_bazel_rules_go//go/platform:windows_amd64": [
            "popcnt_amd64.go",
            "popcnt_amd64.s",
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
