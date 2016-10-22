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
        "cluster.go",
        "doc.go",
        "federation_client.go",
        "generated_expansion.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//federation/apis/federation:go_default_library",
        "//pkg/api:go_default_library",
        "//pkg/apimachinery/registered:go_default_library",
        "//pkg/client/restclient:go_default_library",
        "//pkg/watch:go_default_library",
    ],
)
