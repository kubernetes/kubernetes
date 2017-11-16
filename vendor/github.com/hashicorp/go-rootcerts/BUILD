load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "rootcerts.go",
        "rootcerts_base.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:darwin_amd64": [
            "rootcerts_darwin.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "github.com/hashicorp/go-rootcerts",
    visibility = ["//visibility:public"],
    deps = select({
        "@io_bazel_rules_go//go/platform:darwin_amd64": [
            "//vendor/github.com/mitchellh/go-homedir:go_default_library",
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
