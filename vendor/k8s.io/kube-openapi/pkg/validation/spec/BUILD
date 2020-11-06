load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "external_docs.go",
        "header.go",
        "info.go",
        "ref.go",
        "schema.go",
        "swagger.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/kube-openapi/pkg/validation/spec",
    importpath = "k8s.io/kube-openapi/pkg/validation/spec",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/go-openapi/jsonpointer:go_default_library",
        "//vendor/github.com/go-openapi/jsonreference:go_default_library",
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
