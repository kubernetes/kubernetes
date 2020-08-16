load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "clientset.go",
        "doc.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/kube-aggregator/pkg/client/clientset_generated/deprecated",
    importpath = "k8s.io/kube-aggregator/pkg/client/clientset_generated/deprecated",
    visibility = [":__subpackages__"],
    deps = [
        "//staging/src/k8s.io/client-go/discovery:go_default_library",
        "//staging/src/k8s.io/client-go/rest:go_default_library",
        "//staging/src/k8s.io/client-go/util/flowcontrol:go_default_library",
        "//staging/src/k8s.io/kube-aggregator/pkg/client/clientset_generated/deprecated/typed/apiregistration/v1:go_default_library",
        "//staging/src/k8s.io/kube-aggregator/pkg/client/clientset_generated/deprecated/typed/apiregistration/v1beta1:go_default_library",
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
        "//staging/src/k8s.io/kube-aggregator/pkg/client/clientset_generated/deprecated/fake:all-srcs",
        "//staging/src/k8s.io/kube-aggregator/pkg/client/clientset_generated/deprecated/scheme:all-srcs",
        "//staging/src/k8s.io/kube-aggregator/pkg/client/clientset_generated/deprecated/typed/apiregistration/v1:all-srcs",
        "//staging/src/k8s.io/kube-aggregator/pkg/client/clientset_generated/deprecated/typed/apiregistration/v1beta1:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
