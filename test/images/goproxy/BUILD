package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
)

go_binary(
    name = "goproxy",
    importpath = "k8s.io/kubernetes/test/images/goproxy",
    library = ":go_default_library",
)

go_library(
    name = "go_default_library",
    srcs = ["goproxy.go"],
    importpath = "k8s.io/kubernetes/test/images/goproxy",
    deps = ["//vendor/github.com/elazarl/goproxy:go_default_library"],
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
)
