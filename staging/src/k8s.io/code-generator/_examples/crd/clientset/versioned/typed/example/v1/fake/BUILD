load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "fake_clustertesttype.go",
        "fake_example_client.go",
        "fake_testtype.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/code-generator/_examples/crd/clientset/versioned/typed/example/v1/fake",
    importpath = "k8s.io/code-generator/_examples/crd/clientset/versioned/typed/example/v1/fake",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/api/autoscaling/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/labels:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/types:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/watch:go_default_library",
        "//staging/src/k8s.io/client-go/rest:go_default_library",
        "//staging/src/k8s.io/client-go/testing:go_default_library",
        "//staging/src/k8s.io/code-generator/_examples/crd/apis/example/v1:go_default_library",
        "//staging/src/k8s.io/code-generator/_examples/crd/clientset/versioned/typed/example/v1:go_default_library",
    ],
)
