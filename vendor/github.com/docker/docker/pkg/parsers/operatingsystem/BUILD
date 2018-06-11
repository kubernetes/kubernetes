load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = select({
        "@io_bazel_rules_go//go/platform:darwin": [
            "operatingsystem_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "operatingsystem_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "operatingsystem_linux.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "operatingsystem_solaris.go",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "operatingsystem_windows.go",
        ],
        "//conditions:default": [],
    }),
    cgo = True,
    importpath = "github.com/docker/docker/pkg/parsers/operatingsystem",
    visibility = ["//visibility:public"],
    deps = select({
        "@io_bazel_rules_go//go/platform:linux": [
            "//vendor/github.com/mattn/go-shellwords:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "//vendor/golang.org/x/sys/windows:go_default_library",
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
