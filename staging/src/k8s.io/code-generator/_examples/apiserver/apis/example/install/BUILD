load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["install.go"],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/code-generator/_examples/apiserver/apis/example/install",
    importpath = "k8s.io/code-generator/_examples/apiserver/apis/example/install",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/runtime:go_default_library",
        "//staging/src/k8s.io/code-generator/_examples/apiserver/apis/example:go_default_library",
        "//staging/src/k8s.io/code-generator/_examples/apiserver/apis/example/v1:go_default_library",
    ],
)
