load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "poly1305.go",
        "sum_ref.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:darwin_amd64": [
            "sum_amd64.go",
            "sum_amd64.s",
        ],
        "@io_bazel_rules_go//go/platform:linux_amd64": [
            "sum_amd64.go",
            "sum_amd64.s",
        ],
        "@io_bazel_rules_go//go/platform:windows_amd64": [
            "sum_amd64.go",
            "sum_amd64.s",
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
