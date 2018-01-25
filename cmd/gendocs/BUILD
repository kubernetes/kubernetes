package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
)

go_binary(
    name = "gendocs",
    embed = [":go_default_library"],
    importpath = "k8s.io/kubernetes/cmd/gendocs",
)

go_library(
    name = "go_default_library",
    srcs = ["gen_kubectl_docs.go"],
    importpath = "k8s.io/kubernetes/cmd/gendocs",
    deps = [
        "//cmd/genutils:go_default_library",
        "//pkg/kubectl/cmd:go_default_library",
        "//pkg/kubectl/cmd/util:go_default_library",
        "//vendor/github.com/spf13/cobra/doc:go_default_library",
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
