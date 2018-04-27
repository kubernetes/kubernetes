load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "alt_exit.go",
        "doc.go",
        "entry.go",
        "exported.go",
        "formatter.go",
        "hooks.go",
        "json_formatter.go",
        "logger.go",
        "logrus.go",
        "text_formatter.go",
        "writer.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:darwin": [
            "terminal_bsd.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "terminal_bsd.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "terminal_bsd.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "terminal_linux.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "terminal_bsd.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "terminal_bsd.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "github.com/sirupsen/logrus",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/golang.org/x/crypto/ssh/terminal:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:darwin": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
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
