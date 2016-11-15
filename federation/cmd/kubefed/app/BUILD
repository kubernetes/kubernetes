package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
    "go_test",
    "cgo_library",
)

go_library(
    name = "go_default_library",
    srcs = ["kubefed.go"],
    tags = ["automanaged"],
    deps = [
        "//federation/pkg/kubefed:go_default_library",
        "//pkg/client/metrics/prometheus:go_default_library",
        "//pkg/kubectl/cmd/util:go_default_library",
        "//pkg/util/logs:go_default_library",
        "//pkg/version/prometheus:go_default_library",
    ],
)
