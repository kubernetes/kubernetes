package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
    "go_test",
    "cgo_library",
)

go_binary(
    name = "kubemark",
    srcs = ["hollow-node.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/client/clientset_generated/internalclientset:go_default_library",
        "//pkg/client/metrics/prometheus:go_default_library",
        "//pkg/client/record:go_default_library",
        "//pkg/client/restclient:go_default_library",
        "//pkg/client/unversioned/clientcmd:go_default_library",
        "//pkg/kubelet/cadvisor/testing:go_default_library",
        "//pkg/kubelet/cm:go_default_library",
        "//pkg/kubelet/dockertools:go_default_library",
        "//pkg/kubemark:go_default_library",
        "//pkg/proxy/config:go_default_library",
        "//pkg/util/flag:go_default_library",
        "//pkg/util/iptables/testing:go_default_library",
        "//pkg/util/sets:go_default_library",
        "//pkg/version/prometheus:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/spf13/pflag",
    ],
)
