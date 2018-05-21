load("@io_bazel_rules_go//go:def.bzl", "go_binary", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "goimports.go",
        "goimports_gc.go",
    ],
    importpath = "golang.org/x/tools/cmd/goimports",
    visibility = ["//visibility:private"],
    deps = ["//vendor/golang.org/x/tools/imports:go_default_library"],
)

go_binary(
    name = "goimports",
    embed = [":go_default_library"],
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
