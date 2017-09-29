load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "args.go",
        "delegate.go",
        "exec.go",
        "find.go",
        "raw_exec.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:darwin_amd64": [
            "os_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:linux_amd64": [
            "os_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:windows_amd64": [
            "os_windows.go",
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/containernetworking/cni/pkg/types:go_default_library",
        "//vendor/github.com/containernetworking/cni/pkg/version:go_default_library",
    ],
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
