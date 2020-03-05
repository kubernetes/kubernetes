load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "clustertesttype.go",
        "doc.go",
        "example_client.go",
        "generated_expansion.go",
        "testtype.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/code-generator/_examples/MixedCase/clientset/versioned/typed/example/v1",
    importpath = "k8s.io/code-generator/_examples/MixedCase/clientset/versioned/typed/example/v1",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/api/autoscaling/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/types:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/watch:go_default_library",
        "//staging/src/k8s.io/client-go/rest:go_default_library",
        "//staging/src/k8s.io/code-generator/_examples/MixedCase/apis/example/v1:go_default_library",
        "//staging/src/k8s.io/code-generator/_examples/MixedCase/clientset/versioned/scheme:go_default_library",
    ],
)
