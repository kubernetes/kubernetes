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
        "generator_for_type.go",
        "tags.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor/github.com/golang/glog:go_default_library",
        "//vendor/k8s.io/gengo/args:go_default_library",
        "//vendor/k8s.io/gengo/generator:go_default_library",
        "//vendor/k8s.io/gengo/namer:go_default_library",
        "//vendor/k8s.io/gengo/types:go_default_library",
        "//vendor/k8s.io/kube-gen/cmd/client-gen/args:go_default_library",
        "//vendor/k8s.io/kube-gen/cmd/client-gen/generators/fake:go_default_library",
        "//vendor/k8s.io/kube-gen/cmd/client-gen/generators/scheme:go_default_library",
        "//vendor/k8s.io/kube-gen/cmd/client-gen/generators/util:go_default_library",
        "//vendor/k8s.io/kube-gen/cmd/client-gen/path:go_default_library",
        "//vendor/k8s.io/kube-gen/cmd/client-gen/types:go_default_library",
    ],
)
