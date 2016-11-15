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
    name = "federation-apiserver",
    srcs = ["apiserver.go"],
    tags = ["automanaged"],
    deps = [
        "//federation/cmd/federation-apiserver/app:go_default_library",
        "//federation/cmd/federation-apiserver/app/options:go_default_library",
        "//pkg/util/flag:go_default_library",
        "//pkg/util/logs:go_default_library",
        "//pkg/version/verflag:go_default_library",
        "//vendor:github.com/spf13/pflag",
    ],
)
