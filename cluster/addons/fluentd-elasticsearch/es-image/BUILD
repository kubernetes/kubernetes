package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
)

go_binary(
    name = "es-image",
    importpath = "k8s.io/kubernetes/cluster/addons/fluentd-elasticsearch/es-image",
    library = ":go_default_library",
)

go_library(
    name = "go_default_library",
    srcs = ["elasticsearch_logging_discovery.go"],
    importpath = "k8s.io/kubernetes/cluster/addons/fluentd-elasticsearch/es-image",
    deps = [
        "//pkg/apis/core:go_default_library",
        "//pkg/client/clientset_generated/internalclientset:go_default_library",
        "//vendor/github.com/golang/glog:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/client-go/rest:go_default_library",
        "//vendor/k8s.io/client-go/tools/clientcmd:go_default_library",
        "//vendor/k8s.io/client-go/tools/clientcmd/api:go_default_library",
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
    srcs = [":package-srcs"],
    tags = ["automanaged"],
)
