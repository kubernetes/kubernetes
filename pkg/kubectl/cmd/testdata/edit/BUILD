load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
)

go_binary(
    name = "edit",
    library = ":go_default_library",
    visibility = ["//visibility:public"],
)

go_library(
    name = "go_default_library",
    srcs = ["record.go"],
    visibility = ["//visibility:private"],
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
    visibility = [
        "//build/visible_to:pkg_kubectl_cmd_testdata_edit_CONSUMERS",
    ],
)
