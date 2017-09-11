load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "formatters.go",
        "glog_formatter.go",
        "init.go",
        "journald_formatter.go",
        "log_hijack.go",
        "logmap.go",
        "pkg_logger.go",
        "syslog_formatter.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:windows_amd64": [
            "init_windows.go",
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    deps = ["//vendor/github.com/coreos/go-systemd/journal:go_default_library"],
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
