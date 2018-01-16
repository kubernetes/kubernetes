package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
)
load("//pkg/version:def.bzl", "version_x_defs")

go_binary(
    name = "kubelet",
    embed = [":go_default_library"],
    importpath = "k8s.io/kubernetes/cmd/kubelet",
    x_defs = version_x_defs(),
)

go_library(
    name = "go_default_library",
    srcs = ["kubelet.go"],
    importpath = "k8s.io/kubernetes/cmd/kubelet",
    deps = [
        "//cmd/kubelet/app:go_default_library",
        "//cmd/kubelet/app/options:go_default_library",
        "//pkg/client/metrics/prometheus:go_default_library",
        "//pkg/version/prometheus:go_default_library",
        "//pkg/version/verflag:go_default_library",
        "//vendor/github.com/golang/glog:go_default_library",
        "//vendor/github.com/spf13/pflag:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/util/feature:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/util/flag:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/util/logs:go_default_library",
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
        "//cmd/kubelet/app:all-srcs",
    ],
    tags = ["automanaged"],
)
