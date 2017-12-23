package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = ["tls_test.go"],
    embed = [":go_default_library"],
    importpath = "k8s.io/apiserver/pkg/storage/storagebackend/factory",
    deps = [
        "//vendor/github.com/coreos/etcd/integration:go_default_library",
        "//vendor/github.com/coreos/etcd/pkg/transport:go_default_library",
        "//vendor/golang.org/x/net/context:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/api/testing:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/serializer:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/apis/example:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/apis/example/v1:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/storage/etcd/testing/testingcert:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/storage/storagebackend:go_default_library",
    ],
)

go_library(
    name = "go_default_library",
    srcs = [
        "etcd2.go",
        "etcd3.go",
        "factory.go",
    ],
    importpath = "k8s.io/apiserver/pkg/storage/storagebackend/factory",
    deps = [
        "//vendor/github.com/coreos/etcd/client:go_default_library",
        "//vendor/github.com/coreos/etcd/clientv3:go_default_library",
        "//vendor/github.com/coreos/etcd/pkg/transport:go_default_library",
        "//vendor/golang.org/x/net/context:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/net:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/storage:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/storage/etcd:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/storage/etcd3:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/storage/storagebackend:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/storage/value:go_default_library",
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
