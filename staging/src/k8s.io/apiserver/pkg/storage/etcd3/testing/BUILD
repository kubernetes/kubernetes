package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "test_server.go",
        "utils.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/apiserver/pkg/storage/etcd3/testing",
    importpath = "k8s.io/apiserver/pkg/storage/etcd3/testing",
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/util/wait:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/storage/etcd3/testing/testingcert:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/storage/storagebackend:go_default_library",
        "//vendor/go.etcd.io/etcd/client:go_default_library",
        "//vendor/go.etcd.io/etcd/clientv3:go_default_library",
        "//vendor/go.etcd.io/etcd/etcdserver:go_default_library",
        "//vendor/go.etcd.io/etcd/etcdserver/api/etcdhttp:go_default_library",
        "//vendor/go.etcd.io/etcd/etcdserver/api/v2http:go_default_library",
        "//vendor/go.etcd.io/etcd/integration:go_default_library",
        "//vendor/go.etcd.io/etcd/pkg/testutil:go_default_library",
        "//vendor/go.etcd.io/etcd/pkg/transport:go_default_library",
        "//vendor/go.etcd.io/etcd/pkg/types:go_default_library",
        "//vendor/go.uber.org/zap:go_default_library",
        "//vendor/k8s.io/klog:go_default_library",
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
        "//staging/src/k8s.io/apiserver/pkg/storage/etcd3/testing/testingcert:all-srcs",
    ],
    tags = ["automanaged"],
)
