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
    name = "kube-scheduler",
    srcs = ["scheduler.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/healthz:go_default_library",
        "//pkg/util/flag:go_default_library",
        "//pkg/util/logs:go_default_library",
        "//pkg/version/verflag:go_default_library",
        "//plugin/cmd/kube-scheduler/app:go_default_library",
        "//plugin/cmd/kube-scheduler/app/options:go_default_library",
        "//vendor:github.com/spf13/pflag",
    ],
)
