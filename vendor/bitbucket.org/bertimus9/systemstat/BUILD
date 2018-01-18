load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "systemstat.go",
        "utils.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "systemstat_ex.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "systemstat_ex.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "systemstat_ex.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "systemstat_ex.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "systemstat_linux.go",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "systemstat_ex.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "systemstat_ex.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "systemstat_ex.go",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "systemstat_ex.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "systemstat_ex.go",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "systemstat_ex.go",
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
