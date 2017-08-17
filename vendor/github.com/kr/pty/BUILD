load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "ioctl.go",
        "pty_unsupported.go",
        "run.go",
        "util.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:darwin_amd64": [
            "ioctl_bsd.go",
            "pty_darwin.go",
            "ztypes_amd64.go",
        ],
        "@io_bazel_rules_go//go/platform:linux_amd64": [
            "pty_linux.go",
            "ztypes_amd64.go",
        ],
        "@io_bazel_rules_go//go/platform:windows_amd64": [
            "ztypes_amd64.go",
        ],
        "//conditions:default": [],
    }),
    cgo = True,
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
