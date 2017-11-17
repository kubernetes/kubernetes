load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "capability.go",
        "enum.go",
        "enum_gen.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:darwin_amd64": [
            "capability_noop.go",
        ],
        "@io_bazel_rules_go//go/platform:linux_amd64": [
            "capability_linux.go",
            "syscall_linux.go",
        ],
        "@io_bazel_rules_go//go/platform:windows_amd64": [
            "capability_noop.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "github.com/syndtr/gocapability/capability",
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
