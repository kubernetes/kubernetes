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
        "clusterrole.go",
        "clusterrolebinding.go",
        "doc.go",
        "generated_expansion.go",
        "rbac_client.go",
        "role.go",
        "rolebinding.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/apimachinery/registered:go_default_library",
        "//pkg/apis/rbac:go_default_library",
        "//pkg/client/restclient:go_default_library",
        "//pkg/watch:go_default_library",
    ],
)
