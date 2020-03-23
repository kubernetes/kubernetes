package(default_visibility = ["//visibility:public"])

load(
    "//build:go.bzl",
    go_binary = "go_binary_conditional_pure",
)
load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_binary(
    name = "kubemark",
    embed = [":go_default_library"],
)

go_library(
    name = "go_default_library",
    srcs = ["hollow-node.go"],
    importpath = "k8s.io/kubernetes/cmd/kubemark",
    deps = [
        "//pkg/api/legacyscheme:go_default_library",
        "//pkg/kubelet/cadvisor/testing:go_default_library",
        "//pkg/kubelet/cm:go_default_library",
        "//pkg/kubelet/remote:go_default_library",
        "//pkg/kubelet/remote/fake:go_default_library",
        "//pkg/kubemark:go_default_library",
        "//pkg/master/ports:go_default_library",
        "//pkg/util/iptables/testing:go_default_library",
        "//pkg/util/sysctl/testing:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/sets:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes:go_default_library",
        "//staging/src/k8s.io/client-go/rest:go_default_library",
        "//staging/src/k8s.io/client-go/tools/clientcmd:go_default_library",
        "//staging/src/k8s.io/client-go/tools/record:go_default_library",
        "//staging/src/k8s.io/component-base/cli/flag:go_default_library",
        "//staging/src/k8s.io/component-base/logs:go_default_library",
        "//staging/src/k8s.io/component-base/metrics/prometheus/restclient:go_default_library",
        "//staging/src/k8s.io/component-base/metrics/prometheus/version:go_default_library",
        "//staging/src/k8s.io/component-base/version:go_default_library",
        "//staging/src/k8s.io/component-base/version/verflag:go_default_library",
        "//vendor/github.com/spf13/cobra:go_default_library",
        "//vendor/github.com/spf13/pflag:go_default_library",
        "//vendor/k8s.io/klog:go_default_library",
        "//vendor/k8s.io/utils/exec/testing:go_default_library",
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
