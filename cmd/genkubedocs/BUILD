package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
    "go_test",
)

go_binary(
    name = "genkubedocs",
    embed = [":go_default_library"],
    importpath = "k8s.io/kubernetes/cmd/genkubedocs",
)

go_library(
    name = "go_default_library",
    srcs = [
        "gen_kube_docs.go",
        "postprocessing.go",
    ],
    importpath = "k8s.io/kubernetes/cmd/genkubedocs",
    deps = [
        "//cmd/cloud-controller-manager/app:go_default_library",
        "//cmd/genutils:go_default_library",
        "//cmd/kube-apiserver/app:go_default_library",
        "//cmd/kube-controller-manager/app:go_default_library",
        "//cmd/kube-proxy/app:go_default_library",
        "//cmd/kube-scheduler/app:go_default_library",
        "//cmd/kubeadm/app/cmd:go_default_library",
        "//cmd/kubelet/app:go_default_library",
        "//vendor/github.com/spf13/cobra:go_default_library",
        "//vendor/github.com/spf13/cobra/doc:go_default_library",
        "//vendor/github.com/spf13/pflag:go_default_library",
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

go_test(
    name = "go_default_test",
    srcs = ["postprocessing_test.go"],
    embed = [":go_default_library"],
    importpath = "k8s.io/kubernetes/cmd/genkubedocs",
)
