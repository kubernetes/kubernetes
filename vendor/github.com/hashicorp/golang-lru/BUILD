load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "2q.go",
        "arc.go",
        "doc.go",
        "lru.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/hashicorp/golang-lru",
    importpath = "github.com/hashicorp/golang-lru",
    visibility = ["//visibility:public"],
    deps = ["//vendor/github.com/hashicorp/golang-lru/simplelru:go_default_library"],
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
        "//vendor/github.com/hashicorp/golang-lru/simplelru:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
