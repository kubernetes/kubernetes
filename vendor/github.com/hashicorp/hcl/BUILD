load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "decoder.go",
        "hcl.go",
        "lex.go",
        "parse.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/hashicorp/hcl",
    importpath = "github.com/hashicorp/hcl",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/hashicorp/hcl/hcl/ast:go_default_library",
        "//vendor/github.com/hashicorp/hcl/hcl/parser:go_default_library",
        "//vendor/github.com/hashicorp/hcl/hcl/token:go_default_library",
        "//vendor/github.com/hashicorp/hcl/json/parser:go_default_library",
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
        "//vendor/github.com/hashicorp/hcl/hcl/ast:all-srcs",
        "//vendor/github.com/hashicorp/hcl/hcl/parser:all-srcs",
        "//vendor/github.com/hashicorp/hcl/hcl/printer:all-srcs",
        "//vendor/github.com/hashicorp/hcl/hcl/scanner:all-srcs",
        "//vendor/github.com/hashicorp/hcl/hcl/strconv:all-srcs",
        "//vendor/github.com/hashicorp/hcl/hcl/token:all-srcs",
        "//vendor/github.com/hashicorp/hcl/json/parser:all-srcs",
        "//vendor/github.com/hashicorp/hcl/json/scanner:all-srcs",
        "//vendor/github.com/hashicorp/hcl/json/token:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
