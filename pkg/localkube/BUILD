package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = ["localkube_test.go"],
    library = ":go_default_library",
    tags = ["automanaged"],
)

go_library(
    name = "go_default_library",
    srcs = [
        "apiserver.go",
        "controller-manager.go",
        "crypto.go",
        "etcd.go",
        "extra_options.go",
        "kubelet.go",
        "localkube.go",
        "proxy.go",
        "scheduler.go",
        "server.go",
        "servers.go",
        "util.go",
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
        "//pkg/apis/componentconfig:go_default_library",
        "//pkg/kubelet/qos:go_default_library",
        "//pkg/storage/storagebackend:go_default_library",
        "//pkg/util/config:go_default_library",
        "//pkg/util/net:go_default_library",
        "//plugin/cmd/kube-scheduler/app:go_default_library",
        "//plugin/cmd/kube-scheduler/app/options:go_default_library",
        "//vendor:github.com/coreos/etcd/etcdserver",
        "//vendor:github.com/coreos/etcd/etcdserver/api/v2http",
        "//vendor:github.com/coreos/etcd/pkg/transport",
        "//vendor:github.com/coreos/etcd/pkg/types",
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/pkg/errors",
    ],
)
