load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "config.go",
        "devicetoken.go",
        "msi.go",
        "persist.go",
        "sender.go",
        "token.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:windows_amd64": [
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
