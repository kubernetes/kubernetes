load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "signal.go",
        "signal_posix.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:windows_amd64": [
            "signal_windows.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "k8s.io/sample-controller/pkg/signals",
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
