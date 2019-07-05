load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "context.go",
        "go17.go",
        "go19.go",
        "pre_go17.go",
        "pre_go19.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/golang.org/x/net/context",
    importpath = "golang.org/x/net/context",
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
    srcs = [
        ":package-srcs",
        "//vendor/golang.org/x/net/context/ctxhttp:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
