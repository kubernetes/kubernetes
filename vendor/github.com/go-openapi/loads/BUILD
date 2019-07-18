load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "spec.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/go-openapi/loads",
    importpath = "github.com/go-openapi/loads",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/go-openapi/analysis:go_default_library",
        "//vendor/github.com/go-openapi/spec:go_default_library",
        "//vendor/github.com/go-openapi/swag:go_default_library",
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
