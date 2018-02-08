load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "flags.go",
        "mount.go",
        "mountinfo.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "flags_unsupported.go",
            "mounter_unsupported.go",
            "mountinfo_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "flags_unsupported.go",
            "mounter_unsupported.go",
            "mountinfo_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "flags_unsupported.go",
            "mounter_unsupported.go",
            "mountinfo_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "flags_freebsd.go",
            "flags_unsupported.go",
            "mounter_freebsd.go",
            "mounter_unsupported.go",
            "mountinfo_freebsd.go",
            "mountinfo_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "flags_linux.go",
            "mounter_linux.go",
            "mountinfo_linux.go",
            "sharedsubtree_linux.go",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "flags_unsupported.go",
            "mounter_unsupported.go",
            "mountinfo_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "flags_unsupported.go",
            "mounter_unsupported.go",
            "mountinfo_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "flags_unsupported.go",
            "mounter_unsupported.go",
            "mountinfo_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "flags_unsupported.go",
            "mounter_unsupported.go",
            "mountinfo_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "flags_unsupported.go",
            "mounter_solaris.go",
            "mounter_unsupported.go",
            "mountinfo_solaris.go",
            "mountinfo_unsupported.go",
            "sharedsubtree_solaris.go",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "flags_unsupported.go",
            "mounter_unsupported.go",
            "mountinfo_windows.go",
        ],
        "//conditions:default": [],
    }),
    cgo = True,
    importpath = "github.com/docker/docker/pkg/mount",
    visibility = ["//visibility:public"],
    deps = select({
        "@io_bazel_rules_go//go/platform:freebsd": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
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
