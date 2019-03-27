load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["apply.go"],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/kubectl/pkg",
    importpath = "k8s.io/kubectl/pkg",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/meta:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/types:go_default_library",
        "//staging/src/k8s.io/client-go/rest:go_default_library",
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
        "//staging/src/k8s.io/kubectl/pkg/cmd/apply:all-srcs",
        "//staging/src/k8s.io/kubectl/pkg/cmd/delete:all-srcs",
        "//staging/src/k8s.io/kubectl/pkg/cmd/testing:all-srcs",
        "//staging/src/k8s.io/kubectl/pkg/cmd/util:all-srcs",
        "//staging/src/k8s.io/kubectl/pkg/cmd/wait:all-srcs",
        "//staging/src/k8s.io/kubectl/pkg/generated:all-srcs",
        "//staging/src/k8s.io/kubectl/pkg/scheme:all-srcs",
        "//staging/src/k8s.io/kubectl/pkg/util:all-srcs",
        "//staging/src/k8s.io/kubectl/pkg/validation:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
