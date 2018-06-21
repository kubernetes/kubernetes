load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "discovery.go",
        "env.go",
        "http.go",
        "vcs.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/golang.org/x/tools/go/vcs",
    importpath = "golang.org/x/tools/go/vcs",
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
