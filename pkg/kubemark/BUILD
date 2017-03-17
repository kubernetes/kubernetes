package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "hollow_kubelet.go",
        "hollow_proxy.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//cmd/kube-proxy/app:go_default_library",
        "//cmd/kube-proxy/app/options:go_default_library",
        "//cmd/kubelet/app:go_default_library",
        "//pkg/api:go_default_library",
        "//pkg/apis/componentconfig:go_default_library",
        "//pkg/apis/componentconfig/v1alpha1:go_default_library",
        "//pkg/client/clientset_generated/clientset:go_default_library",
        "//pkg/client/clientset_generated/internalclientset:go_default_library",
        "//pkg/kubelet:go_default_library",
        "//pkg/kubelet/cadvisor:go_default_library",
        "//pkg/kubelet/cm:go_default_library",
        "//pkg/kubelet/container/testing:go_default_library",
        "//pkg/kubelet/dockertools:go_default_library",
        "//pkg/kubelet/types:go_default_library",
        "//pkg/proxy/config:go_default_library",
        "//pkg/util:go_default_library",
        "//pkg/util/io:go_default_library",
        "//pkg/util/iptables:go_default_library",
        "//pkg/util/mount:go_default_library",
        "//pkg/util/oom:go_default_library",
        "//pkg/volume/empty_dir:go_default_library",
        "//pkg/volume/secret:go_default_library",
        "//test/utils:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/apimachinery/pkg/util/wait",
        "//vendor:k8s.io/client-go/kubernetes/typed/core/v1",
        "//vendor:k8s.io/client-go/pkg/api/v1",
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
