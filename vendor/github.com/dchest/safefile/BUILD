load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "safefile.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "rename.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "rename.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "rename.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "rename.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "rename.go",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "rename.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "rename.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "rename.go",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "rename_nonatomic.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "rename.go",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "rename.go",
            "rename_nonatomic.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "github.com/dchest/safefile",
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
