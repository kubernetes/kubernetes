load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "class.go",
        "context.go",
        "doc.go",
        "nickname.go",
        "options.go",
        "profile.go",
        "profiles.go",
        "tables.go",
        "transformer.go",
        "trieval.go",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/golang.org/x/text/cases:go_default_library",
        "//vendor/golang.org/x/text/language:go_default_library",
        "//vendor/golang.org/x/text/runes:go_default_library",
        "//vendor/golang.org/x/text/secure/bidirule:go_default_library",
        "//vendor/golang.org/x/text/transform:go_default_library",
        "//vendor/golang.org/x/text/unicode/norm:go_default_library",
        "//vendor/golang.org/x/text/width:go_default_library",
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
