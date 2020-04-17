package(default_visibility = ["//visibility:public"])

load("//build:code_generation.bzl", "gen_openapi", "openapi_deps")
load("@io_bazel_rules_go//go:def.bzl", "go_library")

gen_openapi(
    outs = ["zz_generated.openapi.go"],
    include_pkgs = [
        "staging/src/k8s.io/apimachinery/pkg/apis/meta/v1",
        "staging/src/k8s.io/apimachinery/pkg/runtime",
        "staging/src/k8s.io/apimachinery/pkg/version",
        "staging/src/k8s.io/sample-apiserver",
    ],
    output_pkg = "k8s.io/sample-apiserver/pkg/generated/openapi",
)

go_library(
    name = "go_default_library",
    srcs = ["zz_generated.openapi.go"],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/sample-apiserver/pkg/generated/openapi",
    importpath = "k8s.io/sample-apiserver/pkg/generated/openapi",
    visibility = ["//visibility:public"],
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
    visibility = ["//visibility:public"],
)
