load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "caller.go",
        "doc.go",
        "domain.go",
        "domain_helper.go",
        "fs.go",
        "gettext.go",
        "local.go",
        "tr.go",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/chai2010/gettext-go/gettext/mo:go_default_library",
        "//vendor/github.com/chai2010/gettext-go/gettext/plural:go_default_library",
        "//vendor/github.com/chai2010/gettext-go/gettext/po:go_default_library",
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
        "//vendor/github.com/chai2010/gettext-go/gettext/mo:all-srcs",
        "//vendor/github.com/chai2010/gettext-go/gettext/plural:all-srcs",
        "//vendor/github.com/chai2010/gettext-go/gettext/po:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
