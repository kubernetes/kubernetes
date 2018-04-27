load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "formatters.go",
        "glog_formatter.go",
        "log_hijack.go",
        "logmap.go",
        "pkg_logger.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "init.go",
            "journald_formatter.go",
            "syslog_formatter.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "init.go",
            "journald_formatter.go",
            "syslog_formatter.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "init.go",
            "journald_formatter.go",
            "syslog_formatter.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "init.go",
            "journald_formatter.go",
            "syslog_formatter.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "init.go",
            "journald_formatter.go",
            "syslog_formatter.go",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "init.go",
            "journald_formatter.go",
            "syslog_formatter.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "init.go",
            "journald_formatter.go",
            "syslog_formatter.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "init.go",
            "journald_formatter.go",
            "syslog_formatter.go",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "init.go",
            "journald_formatter.go",
            "syslog_formatter.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "init.go",
            "journald_formatter.go",
            "syslog_formatter.go",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "init_windows.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "github.com/coreos/pkg/capnslog",
    visibility = ["//visibility:public"],
    deps = select({
        "@io_bazel_rules_go//go/platform:android": [
            "//vendor/github.com/coreos/go-systemd/journal:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "//vendor/github.com/coreos/go-systemd/journal:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "//vendor/github.com/coreos/go-systemd/journal:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "//vendor/github.com/coreos/go-systemd/journal:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "//vendor/github.com/coreos/go-systemd/journal:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "//vendor/github.com/coreos/go-systemd/journal:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "//vendor/github.com/coreos/go-systemd/journal:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "//vendor/github.com/coreos/go-systemd/journal:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "//vendor/github.com/coreos/go-systemd/journal:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "//vendor/github.com/coreos/go-systemd/journal:go_default_library",
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
