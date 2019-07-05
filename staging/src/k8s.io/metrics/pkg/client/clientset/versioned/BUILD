load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "clientset.go",
        "doc.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/metrics/pkg/client/clientset/versioned",
    importpath = "k8s.io/metrics/pkg/client/clientset/versioned",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/client-go/discovery:go_default_library",
        "//staging/src/k8s.io/client-go/rest:go_default_library",
        "//staging/src/k8s.io/client-go/util/flowcontrol:go_default_library",
        "//staging/src/k8s.io/metrics/pkg/client/clientset/versioned/typed/metrics/v1alpha1:go_default_library",
        "//staging/src/k8s.io/metrics/pkg/client/clientset/versioned/typed/metrics/v1beta1:go_default_library",
    ],
)

filegroup(
    name = "package-srcs",
    srcs = glob(["**"]),
    tags = ["automanaged"],
    visibility = ["//visibility:private"],
)

filegroup(
    name = "all-srcs",
    srcs = [
        ":package-srcs",
        "//staging/src/k8s.io/metrics/pkg/client/clientset/versioned/fake:all-srcs",
        "//staging/src/k8s.io/metrics/pkg/client/clientset/versioned/scheme:all-srcs",
        "//staging/src/k8s.io/metrics/pkg/client/clientset/versioned/typed/metrics/v1alpha1:all-srcs",
        "//staging/src/k8s.io/metrics/pkg/client/clientset/versioned/typed/metrics/v1beta1:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
