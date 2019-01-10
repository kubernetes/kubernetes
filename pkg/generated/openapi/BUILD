package(default_visibility = ["//visibility:public"])

load("//build:code_generation.bzl", "gen_openapi", "openapi_deps")
load("@io_bazel_rules_go//go:def.bzl", "go_library")

gen_openapi(
    outs = ["zz_generated.openapi.go"],
    output_pkg = "k8s.io/kubernetes/pkg/generated/openapi",
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

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "zz_generated.openapi.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/generated/openapi",
    deps = [
        "//staging/src/k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/intstr:go_default_library",
        "//vendor/github.com/go-openapi/spec:go_default_library",
        "//vendor/k8s.io/kube-openapi/pkg/common:go_default_library",
    ],
)
