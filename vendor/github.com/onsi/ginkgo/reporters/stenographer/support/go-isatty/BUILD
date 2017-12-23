load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:darwin": [
            "isatty_bsd.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "isatty_bsd.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "isatty_linux.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "isatty_bsd.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "isatty_bsd.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "isatty_solaris.go",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "isatty_windows.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "github.com/onsi/ginkgo/reporters/stenographer/support/go-isatty",
    visibility = ["//visibility:public"],
    deps = select({
        "@io_bazel_rules_go//go/platform:solaris": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
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
