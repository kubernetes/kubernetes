load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "proc.go",
        "unsupported.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:linux_amd64": [
            "linux.go",
            "setns_linux.go",
            "syscall_linux_64.go",
            "sysconfig.go",
            "xattrs_linux.go",
        ],
        "@io_bazel_rules_go//go/platform:windows_amd64": [
            "sysconfig_notcgo.go",
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
