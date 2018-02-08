load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "ioctl.go",
        "run.go",
        "util.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "pty_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "ioctl_bsd.go",
            "pty_darwin.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "ioctl_bsd.go",
            "pty_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "ioctl_bsd.go",
            "pty_freebsd.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "pty_linux.go",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "pty_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "ioctl_bsd.go",
            "pty_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "ioctl_bsd.go",
            "pty_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "pty_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "pty_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "pty_unsupported.go",
        ],
        "//conditions:default": [],
    }) + select({
        "@io_bazel_rules_go//go/platform:386": [
            "ztypes_386.go",
        ],
        "@io_bazel_rules_go//go/platform:amd64": [
            "ztypes_amd64.go",
        ],
        "@io_bazel_rules_go//go/platform:arm": [
            "ztypes_arm.go",
        ],
        "@io_bazel_rules_go//go/platform:arm64": [
            "ztypes_arm64.go",
        ],
        "@io_bazel_rules_go//go/platform:ppc64": [
            "ztypes_ppc64.go",
        ],
        "@io_bazel_rules_go//go/platform:ppc64le": [
            "ztypes_ppc64le.go",
        ],
        "@io_bazel_rules_go//go/platform:s390x": [
            "ztypes_s390x.go",
        ],
        "//conditions:default": [],
    }) + select({
        "@io_bazel_rules_go//go/platform:freebsd_386": [
            "ztypes_freebsd_386.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd_amd64": [
            "ztypes_freebsd_amd64.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd_arm": [
            "ztypes_freebsd_arm.go",
        ],
        "//conditions:default": [],
    }),
    cgo = True,
    importpath = "github.com/kr/pty",
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
