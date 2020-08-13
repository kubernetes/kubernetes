load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "clustertesttype.go",
        "expansion_generated.go",
        "testtype.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/code-generator/_examples/MixedCase/listers/example/v1",
    importpath = "k8s.io/code-generator/_examples/MixedCase/listers/example/v1",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/api/errors:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/labels:go_default_library",
        "//staging/src/k8s.io/client-go/tools/cache:go_default_library",
        "//staging/src/k8s.io/code-generator/_examples/MixedCase/apis/example/v1:go_default_library",
    ],
)
