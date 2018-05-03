load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "osext.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:darwin": [
            "osext_sysctl.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "osext_procfs.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "osext_sysctl.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "osext_procfs.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "osext_procfs.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "osext_procfs.go",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "osext_plan9.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "osext_procfs.go",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "osext_windows.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "github.com/kardianos/osext",
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
