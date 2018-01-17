load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "lookup.go",
        "user.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:darwin": [
            "lookup_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "lookup_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "lookup_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "lookup_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "lookup_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "lookup_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "lookup_unix.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "github.com/opencontainers/runc/libcontainer/user",
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
