load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "config.go",
        "devicetoken.go",
        "persist.go",
        "sender.go",
        "token.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "msi.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "msi.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "msi.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "msi.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "msi.go",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "msi.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "msi.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "msi.go",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "msi.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "msi.go",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "msi_windows.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "github.com/Azure/go-autorest/autorest/adal",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/Azure/go-autorest/autorest/date:go_default_library",
        "//vendor/github.com/dgrijalva/jwt-go:go_default_library",
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
