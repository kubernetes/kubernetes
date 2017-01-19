package(default_visibility = ["//visibility:public"])

licenses(["notice"])

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
    tags = ["automanaged"],
    deps = [
        "//cmd/kube-aggregator/pkg/client/clientset_generated/clientset/typed/apiregistration/v1alpha1:go_default_library",
        "//pkg/client/typed/discovery:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:k8s.io/client-go/pkg/util/flowcontrol",
        "//vendor:k8s.io/client-go/plugin/pkg/client/auth",
        "//vendor:k8s.io/client-go/rest",
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
        "//cmd/kube-aggregator/pkg/client/clientset_generated/clientset/fake:all-srcs",
        "//cmd/kube-aggregator/pkg/client/clientset_generated/clientset/typed/apiregistration/v1alpha1:all-srcs",
    ],
    tags = ["automanaged"],
)
