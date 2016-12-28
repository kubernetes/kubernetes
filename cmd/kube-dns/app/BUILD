package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = ["server.go"],
    tags = ["automanaged"],
    deps = [
        "//cmd/kube-dns/app/options:go_default_library",
        "//pkg/dns:go_default_library",
        "//pkg/dns/config:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/skynetservices/skydns/metrics",
        "//vendor:github.com/skynetservices/skydns/server",
        "//vendor:github.com/spf13/pflag",
        "//vendor:k8s.io/client-go/kubernetes",
        "//vendor:k8s.io/client-go/rest",
        "//vendor:k8s.io/client-go/tools/clientcmd",
    ],
)
