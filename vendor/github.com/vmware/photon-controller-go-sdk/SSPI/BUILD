load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = select({
        "@io_bazel_rules_go//go/platform:android": [
            "sspi_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "sspi_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "sspi_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "sspi_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "sspi_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "sspi_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "sspi_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "sspi_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "sspi_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "sspi_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "sspi.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "github.com/vmware/photon-controller-go-sdk/SSPI",
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
