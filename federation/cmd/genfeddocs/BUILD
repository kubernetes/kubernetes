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
    name = "genfeddocs",
    srcs = ["gen_fed_docs.go"],
    tags = ["automanaged"],
    deps = [
        "//cmd/genutils:go_default_library",
        "//federation/cmd/federation-apiserver/app:go_default_library",
        "//federation/cmd/federation-controller-manager/app:go_default_library",
        "//vendor:github.com/spf13/cobra/doc",
    ],
)
