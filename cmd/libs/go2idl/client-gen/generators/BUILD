package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "client_generator.go",
        "generator_for_clientset.go",
        "generator_for_expansion.go",
        "generator_for_group.go",
        "generator_for_scheme.go",
        "generator_for_type.go",
        "tags.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//cmd/libs/go2idl/client-gen/args:go_default_library",
        "//cmd/libs/go2idl/client-gen/generators/fake:go_default_library",
        "//cmd/libs/go2idl/client-gen/types:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:k8s.io/gengo/args",
        "//vendor:k8s.io/gengo/generator",
        "//vendor:k8s.io/gengo/namer",
        "//vendor:k8s.io/gengo/types",
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
        "//cmd/libs/go2idl/client-gen/generators/fake:all-srcs",
    ],
    tags = ["automanaged"],
)
