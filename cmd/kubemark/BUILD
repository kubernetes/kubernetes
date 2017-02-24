package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
)

go_binary(
    name = "kubemark",
    library = ":go_default_library",
    tags = ["automanaged"],
)

go_library(
    name = "go_default_library",
    srcs = ["hollow-node.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/client/clientset_generated/clientset:go_default_library",
        "//pkg/client/clientset_generated/internalclientset:go_default_library",
        "//pkg/client/metrics/prometheus:go_default_library",
        "//pkg/kubelet/cadvisor/testing:go_default_library",
        "//pkg/kubelet/cm:go_default_library",
        "//pkg/kubelet/dockertools:go_default_library",
        "//pkg/kubemark:go_default_library",
        "//pkg/proxy/config:go_default_library",
        "//pkg/util/iptables/testing:go_default_library",
        "//pkg/version/prometheus:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/spf13/pflag",
        "//vendor:k8s.io/apimachinery/pkg/util/sets",
        "//vendor:k8s.io/apiserver/pkg/util/flag",
        "//vendor:k8s.io/client-go/kubernetes",
        "//vendor:k8s.io/client-go/pkg/api/v1",
        "//vendor:k8s.io/client-go/rest",
        "//vendor:k8s.io/client-go/tools/clientcmd",
        "//vendor:k8s.io/client-go/tools/record",
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
