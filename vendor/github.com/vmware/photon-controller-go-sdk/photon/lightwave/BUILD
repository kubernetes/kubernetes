load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "jwttoken.go",
        "oidcclient.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "oidcclient_sspi_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "oidcclient_sspi_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "oidcclient_sspi_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "oidcclient_sspi_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "oidcclient_sspi_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "oidcclient_sspi_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "oidcclient_sspi_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "oidcclient_sspi_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "oidcclient_sspi_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "oidcclient_sspi_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "oidcclient_sspi.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "github.com/vmware/photon-controller-go-sdk/photon/lightwave",
    visibility = ["//visibility:public"],
    deps = select({
        "@io_bazel_rules_go//go/platform:windows": [
            "//vendor/github.com/vmware/photon-controller-go-sdk/SSPI:go_default_library",
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
