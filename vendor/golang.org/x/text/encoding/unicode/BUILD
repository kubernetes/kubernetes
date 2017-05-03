package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "override.go",
        "unicode.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor/golang.org/x/text/encoding:go_default_library",
        "//vendor/golang.org/x/text/encoding/internal:go_default_library",
        "//vendor/golang.org/x/text/encoding/internal/identifier:go_default_library",
        "//vendor/golang.org/x/text/internal/utf8internal:go_default_library",
        "//vendor/golang.org/x/text/runes:go_default_library",
        "//vendor/golang.org/x/text/transform:go_default_library",
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
)
