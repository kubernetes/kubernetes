package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
)

go_binary(
    name = "genkubedocs",
    library = ":go_default_library",
)

go_library(
    name = "go_default_library",
    srcs = ["gen_kube_docs.go"],
    deps = [
        "//cmd/cloud-controller-manager/app:go_default_library",
        "//cmd/genutils:go_default_library",
        "//cmd/kube-apiserver/app:go_default_library",
        "//cmd/kube-controller-manager/app:go_default_library",
        "//cmd/kube-proxy/app:go_default_library",
        "//cmd/kubelet/app:go_default_library",
        "//plugin/cmd/kube-scheduler/app:go_default_library",
        "//vendor/github.com/spf13/cobra/doc:go_default_library",
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
