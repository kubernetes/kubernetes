package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
    "go_test",
    "cgo_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "autoscaling_client.go",
        "doc.go",
        "generated_expansion.go",
        "horizontalpodautoscaler.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//pkg/apimachinery/registered:go_default_library",
        "//pkg/apis/autoscaling/v1:go_default_library",
        "//pkg/client/restclient:go_default_library",
        "//pkg/runtime/schema:go_default_library",
        "//pkg/runtime/serializer:go_default_library",
        "//pkg/watch:go_default_library",
    ],
)
