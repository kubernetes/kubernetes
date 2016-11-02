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
        "fake_client_generator.go",
        "generator_fake_for_clientset.go",
        "generator_fake_for_group.go",
        "generator_fake_for_type.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//cmd/libs/go2idl/client-gen/args:go_default_library",
        "//cmd/libs/go2idl/client-gen/types:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:k8s.io/gengo/generator",
        "//vendor:k8s.io/gengo/namer",
        "//vendor:k8s.io/gengo/types",
    ],
)
