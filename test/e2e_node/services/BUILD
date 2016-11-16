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
    srcs = [
        "apiserver.go",
        "etcd.go",
        "internal_services.go",
        "kubelet.go",
        "namespace_controller.go",
        "server.go",
        "services.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//cmd/kube-apiserver/app:go_default_library",
        "//cmd/kube-apiserver/app/options:go_default_library",
        "//pkg/api:go_default_library",
        "//pkg/apimachinery/registered:go_default_library",
        "//pkg/client/clientset_generated/internalclientset:go_default_library",
        "//pkg/client/restclient:go_default_library",
        "//pkg/client/typed/dynamic:go_default_library",
        "//pkg/controller/namespace:go_default_library",
        "//pkg/util/config:go_default_library",
        "//test/e2e/framework:go_default_library",
        "//test/e2e_node/builder:go_default_library",
        "//vendor:github.com/coreos/etcd/etcdserver",
        "//vendor:github.com/coreos/etcd/etcdserver/api/v2http",
        "//vendor:github.com/coreos/etcd/pkg/transport",
        "//vendor:github.com/coreos/etcd/pkg/types",
        "//vendor:github.com/coreos/pkg/capnslog",
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/kardianos/osext",
    ],
)
