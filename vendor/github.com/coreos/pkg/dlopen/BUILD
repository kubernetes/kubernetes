load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "dlopen.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:linux_amd64": [
            "dlopen_example.go",
        ],
        "//conditions:default": [],
    }),
    cgo = True,
    clinkopts = ["-ldl"],
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
