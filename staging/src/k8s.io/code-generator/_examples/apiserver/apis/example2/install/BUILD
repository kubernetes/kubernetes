load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["install.go"],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/code-generator/_examples/apiserver/apis/example2/install",
    importpath = "k8s.io/code-generator/_examples/apiserver/apis/example2/install",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/runtime:go_default_library",
        "//staging/src/k8s.io/code-generator/_examples/apiserver/apis/example2:go_default_library",
        "//staging/src/k8s.io/code-generator/_examples/apiserver/apis/example2/v1:go_default_library",
    ],
)
