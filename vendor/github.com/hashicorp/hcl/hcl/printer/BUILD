load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "nodes.go",
        "printer.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/hashicorp/hcl/hcl/printer",
    importpath = "github.com/hashicorp/hcl/hcl/printer",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/hashicorp/hcl/hcl/ast:go_default_library",
        "//vendor/github.com/hashicorp/hcl/hcl/parser:go_default_library",
        "//vendor/github.com/hashicorp/hcl/hcl/token:go_default_library",
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
