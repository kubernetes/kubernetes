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
    name = "hyperkube",
    srcs = [
        "federation-apiserver.go",
        "federation-controller-manager.go",
        "hyperkube.go",
        "kube-apiserver.go",
        "kube-controller-manager.go",
        "kube-proxy.go",
        "kube-scheduler.go",
        "kubectl.go",
        "kubelet.go",
        "main.go",
        "server.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//cmd/kube-apiserver/app:go_default_library",
        "//cmd/kube-apiserver/app/options:go_default_library",
        "//cmd/kube-controller-manager/app:go_default_library",
        "//cmd/kube-controller-manager/app/options:go_default_library",
        "//cmd/kube-proxy/app:go_default_library",
        "//cmd/kube-proxy/app/options:go_default_library",
        "//cmd/kubelet/app:go_default_library",
        "//cmd/kubelet/app/options:go_default_library",
        "//federation/cmd/federation-apiserver/app:go_default_library",
        "//federation/cmd/federation-apiserver/app/options:go_default_library",
        "//federation/cmd/federation-controller-manager/app:go_default_library",
        "//federation/cmd/federation-controller-manager/app/options:go_default_library",
        "//pkg/client/metrics/prometheus:go_default_library",
        "//pkg/healthz:go_default_library",
        "//pkg/kubectl/cmd:go_default_library",
        "//pkg/kubectl/cmd/util:go_default_library",
        "//pkg/util:go_default_library",
        "//pkg/util/flag:go_default_library",
        "//pkg/util/logs:go_default_library",
        "//pkg/version/prometheus:go_default_library",
        "//pkg/version/verflag:go_default_library",
        "//plugin/cmd/kube-scheduler/app:go_default_library",
        "//plugin/cmd/kube-scheduler/app/options:go_default_library",
        "//vendor:github.com/spf13/pflag",
    ],
)
