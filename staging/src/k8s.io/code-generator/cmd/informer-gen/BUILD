package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
)

go_binary(
    name = "informer-gen",
    embed = [":go_default_library"],
    importpath = "k8s.io/code-generator/cmd/informer-gen",
)

go_library(
    name = "go_default_library",
    srcs = ["main.go"],
    importpath = "k8s.io/code-generator/cmd/informer-gen",
    deps = [
        "//vendor/github.com/golang/glog:go_default_library",
        "//vendor/github.com/spf13/pflag:go_default_library",
        "//vendor/k8s.io/code-generator/cmd/informer-gen/args:go_default_library",
        "//vendor/k8s.io/code-generator/cmd/informer-gen/generators:go_default_library",
        "//vendor/k8s.io/gengo/args:go_default_library",
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
        "//staging/src/k8s.io/code-generator/cmd/informer-gen/args:all-srcs",
        "//staging/src/k8s.io/code-generator/cmd/informer-gen/generators:all-srcs",
    ],
    tags = ["automanaged"],
)
