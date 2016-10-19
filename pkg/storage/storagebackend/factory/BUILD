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
        "etcd2.go",
        "etcd3.go",
        "factory.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/storage:go_default_library",
        "//pkg/storage/etcd:go_default_library",
        "//pkg/storage/etcd3:go_default_library",
        "//pkg/storage/storagebackend:go_default_library",
        "//pkg/util/net:go_default_library",
        "//vendor:github.com/coreos/etcd/client",
        "//vendor:github.com/coreos/etcd/clientv3",
        "//vendor:github.com/coreos/etcd/pkg/transport",
        "//vendor:golang.org/x/net/context",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["tls_test.go"],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/testapi:go_default_library",
        "//pkg/storage/etcd/testing/testingcert:go_default_library",
        "//pkg/storage/storagebackend:go_default_library",
        "//vendor:github.com/coreos/etcd/integration",
        "//vendor:github.com/coreos/etcd/pkg/transport",
        "//vendor:golang.org/x/net/context",
    ],
)
