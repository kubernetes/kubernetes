load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "bucket.go",
        "cursor.go",
        "db.go",
        "doc.go",
        "errors.go",
        "freelist.go",
        "node.go",
        "page.go",
        "tx.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "bolt_unix.go",
            "boltsync_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "bolt_unix.go",
            "boltsync_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "bolt_unix.go",
            "boltsync_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "bolt_unix.go",
            "boltsync_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "bolt_linux.go",
            "bolt_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "bolt_unix.go",
            "boltsync_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "bolt_unix.go",
            "boltsync_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "bolt_openbsd.go",
            "bolt_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "bolt_unix_solaris.go",
            "boltsync_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "bolt_windows.go",
        ],
        "//conditions:default": [],
    }) + select({
        "@io_bazel_rules_go//go/platform:386": [
            "bolt_386.go",
        ],
        "@io_bazel_rules_go//go/platform:amd64": [
            "bolt_amd64.go",
        ],
        "@io_bazel_rules_go//go/platform:arm": [
            "bolt_arm.go",
        ],
        "@io_bazel_rules_go//go/platform:arm64": [
            "bolt_arm64.go",
        ],
        "@io_bazel_rules_go//go/platform:mips": [
            "bolt_mipsx.go",
        ],
        "@io_bazel_rules_go//go/platform:mips64": [
            "bolt_mips64x.go",
        ],
        "@io_bazel_rules_go//go/platform:mips64le": [
            "bolt_mips64x.go",
        ],
        "@io_bazel_rules_go//go/platform:mipsle": [
            "bolt_mipsx.go",
        ],
        "@io_bazel_rules_go//go/platform:ppc64": [
            "bolt_ppc64.go",
        ],
        "@io_bazel_rules_go//go/platform:ppc64le": [
            "bolt_ppc64le.go",
        ],
        "@io_bazel_rules_go//go/platform:s390x": [
            "bolt_s390x.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "github.com/coreos/bbolt",
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
