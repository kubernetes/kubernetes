package(default_visibility = ["//visibility:public"])

load("//build:code_generation.bzl", "gen_openapi", "openapi_deps")
load("@io_bazel_rules_go//go:def.bzl", "go_library")

gen_openapi(
    outs = ["zz_generated.openapi.go"],
    output_pkg = "k8s.io/apiextensions-apiserver/pkg/generated/openapi",
)

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "zz_generated.openapi.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/apiextensions-apiserver/pkg/generated/openapi",
    importpath = "k8s.io/apiextensions-apiserver/pkg/generated/openapi",
    deps = openapi_deps(),  # keep
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
)
