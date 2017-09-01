load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "entry.go",
        "exported.go",
        "formatter.go",
        "hooks.go",
        "json_formatter.go",
        "logger.go",
        "logrus.go",
        "text_formatter.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:darwin_amd64": [
            "terminal_darwin.go",
            "terminal_notwindows.go",
        ],
        "@io_bazel_rules_go//go/platform:linux_amd64": [
            "terminal_linux.go",
            "terminal_notwindows.go",
        ],
        "@io_bazel_rules_go//go/platform:windows_amd64": [
            "terminal_windows.go",
        ],
        "//conditions:default": [],
    }),
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
