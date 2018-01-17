load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "proc.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "sysconfig_notcgo.go",
            "unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "sysconfig_notcgo.go",
            "unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "sysconfig_notcgo.go",
            "unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "sysconfig_notcgo.go",
            "unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "linux.go",
            "sysconfig.go",
            "sysconfig_notcgo.go",
            "xattrs_linux.go",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "sysconfig_notcgo.go",
            "unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "sysconfig_notcgo.go",
            "unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "sysconfig_notcgo.go",
            "unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "sysconfig_notcgo.go",
            "unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "sysconfig_notcgo.go",
            "unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "sysconfig_notcgo.go",
            "unsupported.go",
        ],
        "//conditions:default": [],
    }) + select({
        "@io_bazel_rules_go//go/platform:linux_386": [
            "syscall_linux_32.go",
        ],
        "@io_bazel_rules_go//go/platform:linux_amd64": [
            "syscall_linux_64.go",
        ],
        "@io_bazel_rules_go//go/platform:linux_arm": [
            "syscall_linux_32.go",
        ],
        "@io_bazel_rules_go//go/platform:linux_arm64": [
            "syscall_linux_64.go",
        ],
        "@io_bazel_rules_go//go/platform:linux_mips": [
            "syscall_linux_64.go",
        ],
        "@io_bazel_rules_go//go/platform:linux_mips64": [
            "syscall_linux_64.go",
        ],
        "@io_bazel_rules_go//go/platform:linux_mips64le": [
            "syscall_linux_64.go",
        ],
        "@io_bazel_rules_go//go/platform:linux_mipsle": [
            "syscall_linux_64.go",
        ],
        "@io_bazel_rules_go//go/platform:linux_ppc64": [
            "syscall_linux_64.go",
        ],
        "@io_bazel_rules_go//go/platform:linux_ppc64le": [
            "syscall_linux_64.go",
        ],
        "@io_bazel_rules_go//go/platform:linux_s390x": [
            "syscall_linux_64.go",
        ],
        "//conditions:default": [],
    }),
    cgo = True,
    importpath = "github.com/opencontainers/runc/libcontainer/system",
    visibility = ["//visibility:public"],
    deps = select({
        "@io_bazel_rules_go//go/platform:linux": [
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
