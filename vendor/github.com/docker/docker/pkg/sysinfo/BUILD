load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "sysinfo.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "numcpu.go",
            "sysinfo_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "numcpu.go",
            "sysinfo_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "numcpu.go",
            "sysinfo_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "numcpu.go",
            "sysinfo_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "numcpu_linux.go",
            "sysinfo_linux.go",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "numcpu.go",
            "sysinfo_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "numcpu.go",
            "sysinfo_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "numcpu.go",
            "sysinfo_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "numcpu.go",
            "sysinfo_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "numcpu.go",
            "sysinfo_solaris.go",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "numcpu_windows.go",
            "sysinfo_windows.go",
        ],
        "//conditions:default": [],
    }),
    cgo = True,
    clinkopts = select({
        "@io_bazel_rules_go//go/platform:solaris": [
            "-llgrp",
        ],
        "//conditions:default": [],
    }),
    importpath = "github.com/docker/docker/pkg/sysinfo",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/docker/docker/pkg/parsers:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:linux": [
            "//vendor/github.com/opencontainers/runc/libcontainer/cgroups:go_default_library",
            "//vendor/github.com/sirupsen/logrus:go_default_library",
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
