load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "noncolorable.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "colorable_others.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "colorable_others.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "colorable_others.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "colorable_others.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "colorable_others.go",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "colorable_others.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "colorable_others.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "colorable_others.go",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "colorable_others.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "colorable_others.go",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "colorable_windows.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "github.com/onsi/ginkgo/reporters/stenographer/support/go-colorable",
    visibility = ["//visibility:public"],
    deps = select({
        "@io_bazel_rules_go//go/platform:windows": [
            "//vendor/github.com/onsi/ginkgo/reporters/stenographer/support/go-isatty:go_default_library",
        ],
        "//conditions:default": [],
    }),
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
