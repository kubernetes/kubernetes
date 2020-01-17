load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["zz_generated.openapi.go"],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/code-generator/_examples/apiserver/openapi",
    importpath = "k8s.io/code-generator/_examples/apiserver/openapi",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/github.com/go-openapi/spec:go_default_library",
        "//vendor/k8s.io/kube-openapi/pkg/common:go_default_library",
    ],
)
