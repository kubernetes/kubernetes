load("@io_bazel_rules_go//go:def.bzl", "go_binary", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["cfssljson.go"],
    importmap = "k8s.io/kubernetes/vendor/github.com/cloudflare/cfssl/cmd/cfssljson",
    importpath = "github.com/cloudflare/cfssl/cmd/cfssljson",
    visibility = ["//visibility:private"],
    deps = ["//vendor/github.com/cloudflare/cfssl/cli/version:go_default_library"],
)

go_binary(
    name = "cfssljson",
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
