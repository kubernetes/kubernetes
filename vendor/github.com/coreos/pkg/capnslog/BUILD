load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "formatters.go",
        "glog_formatter.go",
        "init.go",
        "init_windows.go",
        "journald_formatter.go",
        "log_hijack.go",
        "logmap.go",
        "pkg_logger.go",
        "syslog_formatter.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/coreos/pkg/capnslog",
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
        "@io_bazel_rules_go//go/platform:ios": [
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
