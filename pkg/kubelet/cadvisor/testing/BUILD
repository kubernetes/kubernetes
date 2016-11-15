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
    srcs = [
        "cadvisor_fake.go",
        "cadvisor_mock.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/kubelet/cadvisor:go_default_library",
        "//vendor:github.com/google/cadvisor/events",
        "//vendor:github.com/google/cadvisor/info/v1",
        "//vendor:github.com/google/cadvisor/info/v2",
        "//vendor:github.com/stretchr/testify/mock",
    ],
)
