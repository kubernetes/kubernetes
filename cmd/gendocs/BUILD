package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
    "go_test",
    "cgo_library",
)

go_binary(
    name = "gendocs",
    srcs = ["gen_kubectl_docs.go"],
    tags = ["automanaged"],
    deps = [
        "//cmd/genutils:go_default_library",
        "//pkg/kubectl/cmd:go_default_library",
        "//pkg/kubectl/cmd/util:go_default_library",
        "//vendor:github.com/spf13/cobra/doc",
    ],
)
