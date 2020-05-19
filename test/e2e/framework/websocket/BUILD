load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["websocket_util.go"],
    importpath = "k8s.io/kubernetes/test/e2e/framework/websocket",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/client-go/rest:go_default_library",
        "//vendor/golang.org/x/net/websocket:go_default_library",
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
