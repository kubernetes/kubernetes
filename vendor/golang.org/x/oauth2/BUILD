package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "oauth2.go",
        "token.go",
        "transport.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor/golang.org/x/net/context:go_default_library",
        "//vendor/golang.org/x/oauth2/internal:go_default_library",
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
    srcs = [
        ":package-srcs",
        "//vendor/golang.org/x/oauth2/google:all-srcs",
        "//vendor/golang.org/x/oauth2/internal:all-srcs",
        "//vendor/golang.org/x/oauth2/jws:all-srcs",
        "//vendor/golang.org/x/oauth2/jwt:all-srcs",
    ],
    tags = ["automanaged"],
)
