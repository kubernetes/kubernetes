package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
    "go_test",
    "cgo_library",
)

go_library(
    name = "go_default_library",
    srcs = ["server.go"],
    tags = ["automanaged"],
    deps = [
        "//cmd/kube-dns/app/options:go_default_library",
        "//pkg/client/clientset_generated/release_1_5:go_default_library",
        "//pkg/client/restclient:go_default_library",
        "//pkg/client/unversioned/clientcmd:go_default_library",
        "//pkg/dns:go_default_library",
        "//pkg/dns/config:go_default_library",
        "//pkg/runtime/schema:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/skynetservices/skydns/metrics",
        "//vendor:github.com/skynetservices/skydns/server",
        "//vendor:github.com/spf13/pflag",
    ],
)
