load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "tokenreview.go",
        "tokenreviewspec.go",
        "tokenreviewstatus.go",
        "userinfo.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/client-go/applyconfigurations/authentication/v1beta1",
    importpath = "k8s.io/client-go/applyconfigurations/authentication/v1beta1",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/api/authentication/v1beta1:go_default_library",
        "//staging/src/k8s.io/client-go/applyconfigurations/meta/v1:go_default_library",
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
