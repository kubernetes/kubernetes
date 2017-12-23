load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "terminal.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:darwin": [
            "util.go",
            "util_bsd.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "util.go",
            "util_bsd.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "util.go",
            "util_bsd.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "util.go",
            "util_linux.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "util.go",
            "util_bsd.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "util.go",
            "util_bsd.go",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "util_plan9.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "util_solaris.go",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "util_windows.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "golang.org/x/crypto/ssh/terminal",
    visibility = ["//visibility:public"],
    deps = select({
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
        "@io_bazel_rules_go//go/platform:solaris": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "//vendor/golang.org/x/sys/windows:go_default_library",
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
