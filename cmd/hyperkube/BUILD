package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
    "go_test",
)
load("//pkg/version:def.bzl", "version_x_defs")

go_binary(
    name = "hyperkube",
    library = ":go_default_library",
    x_defs = version_x_defs(),
)

go_test(
    name = "go_default_test",
    srcs = ["hyperkube_test.go"],
    library = ":go_default_library",
    deps = [
        "//vendor/github.com/spf13/cobra:go_default_library",
        "//vendor/github.com/stretchr/testify/assert:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/wait:go_default_library",
    ],
)

go_library(
    name = "go_default_library",
    srcs = [
        "federation-apiserver.go",
        "federation-controller-manager.go",
        "hyperkube.go",
        "kube-aggregator.go",
        "kube-apiserver.go",
        "kube-controller-manager.go",
        "kube-proxy.go",
        "kube-scheduler.go",
        "kubectl.go",
        "kubelet.go",
        "main.go",
        "server.go",
    ],
    deps = [
        "//cmd/kube-apiserver/app:go_default_library",
        "//cmd/kube-apiserver/app/options:go_default_library",
        "//cmd/kube-controller-manager/app:go_default_library",
        "//cmd/kube-controller-manager/app/options:go_default_library",
        "//cmd/kube-proxy/app:go_default_library",
        "//cmd/kubelet/app:go_default_library",
        "//cmd/kubelet/app/options:go_default_library",
        "//federation/cmd/federation-apiserver/app:go_default_library",
        "//federation/cmd/federation-apiserver/app/options:go_default_library",
        "//federation/cmd/federation-controller-manager/app:go_default_library",
        "//federation/cmd/federation-controller-manager/app/options:go_default_library",
        "//pkg/client/metrics/prometheus:go_default_library",
        "//pkg/kubectl/cmd:go_default_library",
        "//pkg/kubectl/cmd/util:go_default_library",
        "//pkg/util/template:go_default_library",
        "//pkg/version/prometheus:go_default_library",
        "//pkg/version/verflag:go_default_library",
        "//plugin/cmd/kube-scheduler/app:go_default_library",
        "//plugin/cmd/kube-scheduler/app/options:go_default_library",
        "//vendor/github.com/spf13/pflag:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/wait:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/server:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/server/healthz:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/util/flag:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/util/logs:go_default_library",
        "//vendor/k8s.io/kube-aggregator/pkg/cmd/server:go_default_library",
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
