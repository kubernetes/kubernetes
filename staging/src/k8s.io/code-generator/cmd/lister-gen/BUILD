package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
)

go_binary(
    name = "lister-gen",
    embed = [":go_default_library"],
)

go_library(
    name = "go_default_library",
    srcs = ["main.go"],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/code-generator/cmd/lister-gen",
    importpath = "k8s.io/code-generator/cmd/lister-gen",
    deps = [
        "//staging/src/k8s.io/code-generator/cmd/lister-gen/args:go_default_library",
        "//staging/src/k8s.io/code-generator/cmd/lister-gen/generators:go_default_library",
        "//staging/src/k8s.io/code-generator/pkg/util:go_default_library",
        "//vendor/github.com/spf13/pflag:go_default_library",
        "//vendor/k8s.io/gengo/args:go_default_library",
        "//vendor/k8s.io/klog:go_default_library",
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
        "//staging/src/k8s.io/code-generator/cmd/lister-gen/args:all-srcs",
        "//staging/src/k8s.io/code-generator/cmd/lister-gen/generators:all-srcs",
    ],
    tags = ["automanaged"],
)
