load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["testtype.go"],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/code-generator/_examples/apiserver/listers/example/internalversion_fakelister",
    importpath = "k8s.io/code-generator/_examples/apiserver/listers/example/internalversion_fakelister",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/client-go/tools/cache:go_default_library",
        "//staging/src/k8s.io/code-generator/_examples/apiserver/apis/example:go_default_library",
        "//staging/src/k8s.io/code-generator/_examples/apiserver/listers/example/internalversion:go_default_library",
    ],
)
