load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = select({
        "@io_bazel_rules_go//go/platform:android": [
            "apply_nosystemd.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "apply_nosystemd.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "apply_nosystemd.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "apply_nosystemd.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "apply_systemd.go",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "apply_nosystemd.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "apply_nosystemd.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "apply_nosystemd.go",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "apply_nosystemd.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "apply_nosystemd.go",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "apply_nosystemd.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "github.com/opencontainers/runc/libcontainer/cgroups/systemd",
    visibility = ["//visibility:public"],
    deps = select({
        "@io_bazel_rules_go//go/platform:android": [
            "//vendor/github.com/opencontainers/runc/libcontainer/cgroups:go_default_library",
            "//vendor/github.com/opencontainers/runc/libcontainer/configs:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "//vendor/github.com/opencontainers/runc/libcontainer/cgroups:go_default_library",
            "//vendor/github.com/opencontainers/runc/libcontainer/configs:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "//vendor/github.com/opencontainers/runc/libcontainer/cgroups:go_default_library",
            "//vendor/github.com/opencontainers/runc/libcontainer/configs:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "//vendor/github.com/opencontainers/runc/libcontainer/cgroups:go_default_library",
            "//vendor/github.com/opencontainers/runc/libcontainer/configs:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "//vendor/github.com/coreos/go-systemd/dbus:go_default_library",
            "//vendor/github.com/coreos/go-systemd/util:go_default_library",
            "//vendor/github.com/godbus/dbus:go_default_library",
            "//vendor/github.com/opencontainers/runc/libcontainer/cgroups:go_default_library",
            "//vendor/github.com/opencontainers/runc/libcontainer/cgroups/fs:go_default_library",
            "//vendor/github.com/opencontainers/runc/libcontainer/configs:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "//vendor/github.com/opencontainers/runc/libcontainer/cgroups:go_default_library",
            "//vendor/github.com/opencontainers/runc/libcontainer/configs:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "//vendor/github.com/opencontainers/runc/libcontainer/cgroups:go_default_library",
            "//vendor/github.com/opencontainers/runc/libcontainer/configs:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "//vendor/github.com/opencontainers/runc/libcontainer/cgroups:go_default_library",
            "//vendor/github.com/opencontainers/runc/libcontainer/configs:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "//vendor/github.com/opencontainers/runc/libcontainer/cgroups:go_default_library",
            "//vendor/github.com/opencontainers/runc/libcontainer/configs:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "//vendor/github.com/opencontainers/runc/libcontainer/cgroups:go_default_library",
            "//vendor/github.com/opencontainers/runc/libcontainer/configs:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "//vendor/github.com/opencontainers/runc/libcontainer/cgroups:go_default_library",
            "//vendor/github.com/opencontainers/runc/libcontainer/configs:go_default_library",
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
