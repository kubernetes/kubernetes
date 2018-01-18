package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "clientset.go",
        "doc.go",
    ],
    importpath = "k8s.io/metrics/pkg/client/clientset_generated/clientset",
    deps = [
        "//vendor/github.com/golang/glog:go_default_library",
        "//vendor/k8s.io/client-go/discovery:go_default_library",
        "//vendor/k8s.io/client-go/rest:go_default_library",
        "//vendor/k8s.io/client-go/util/flowcontrol:go_default_library",
        "//vendor/k8s.io/metrics/pkg/client/clientset_generated/clientset/typed/metrics/v1alpha1:go_default_library",
        "//vendor/k8s.io/metrics/pkg/client/clientset_generated/clientset/typed/metrics/v1beta1:go_default_library",
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
        "//staging/src/k8s.io/metrics/pkg/client/clientset_generated/clientset/fake:all-srcs",
        "//staging/src/k8s.io/metrics/pkg/client/clientset_generated/clientset/scheme:all-srcs",
        "//staging/src/k8s.io/metrics/pkg/client/clientset_generated/clientset/typed/metrics/v1alpha1:all-srcs",
        "//staging/src/k8s.io/metrics/pkg/client/clientset_generated/clientset/typed/metrics/v1beta1:all-srcs",
    ],
    tags = ["automanaged"],
)
