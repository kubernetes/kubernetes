load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "default_generator.go",
        "default_package.go",
        "doc.go",
        "error_tracker.go",
        "execute.go",
        "generator.go",
        "import_tracker.go",
        "snippet_writer.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/gengo/generator",
    importpath = "k8s.io/gengo/generator",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/golang.org/x/tools/imports:go_default_library",
        "//vendor/k8s.io/gengo/namer:go_default_library",
        "//vendor/k8s.io/gengo/parser:go_default_library",
        "//vendor/k8s.io/gengo/types:go_default_library",
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
    srcs = [":package-srcs"],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
