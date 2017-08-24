package(default_visibility = ["//pkg/kubectl/visibility:kubectl_allowed_deps"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
)

go_binary(
    name = "edit",
    library = ":go_default_library",
)

go_library(
    name = "go_default_library",
    srcs = ["record.go"],
    deps = ["//vendor/gopkg.in/yaml.v2:go_default_library"],
)

filegroup(
    name = "package-srcs",
    srcs = glob(["**"]),
    tags = ["automanaged"],
)

filegroup(
    name = "all-srcs",
    srcs = [":package-srcs"],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
