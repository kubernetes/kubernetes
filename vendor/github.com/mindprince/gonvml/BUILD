load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "bindings.go",
        "nvml.h",
    ],
    cgo = True,
    clinkopts = ["-ldl"],
    importmap = "k8s.io/kubernetes/vendor/github.com/mindprince/gonvml",
    importpath = "github.com/mindprince/gonvml",
    visibility = ["//visibility:public"],
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
