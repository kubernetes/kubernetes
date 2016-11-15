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
    name = "genkubedocs",
    srcs = ["gen_kube_docs.go"],
    tags = ["automanaged"],
    deps = [
        "//cmd/genutils:go_default_library",
        "//cmd/kube-apiserver/app:go_default_library",
        "//cmd/kube-controller-manager/app:go_default_library",
        "//cmd/kube-proxy/app:go_default_library",
        "//cmd/kubelet/app:go_default_library",
        "//plugin/cmd/kube-scheduler/app:go_default_library",
        "//vendor:github.com/spf13/cobra/doc",
    ],
)
