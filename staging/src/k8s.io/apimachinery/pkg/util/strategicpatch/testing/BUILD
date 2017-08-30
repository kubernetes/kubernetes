package(default_visibility = ["//visibility:public"])

load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["openapi.go"],
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/kubectl/cmd/util/openapi:go_default_library",
        "//pkg/kubectl/cmd/util/openapi/testing:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
    ],
)

filegroup(
    name = "testing-openapi-spec",
    srcs = [
        "swagger-merge-item.json",
        "swagger-precision-item.json",
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
    testonly = True,
    visibility = ["//visibility:public"],
)
