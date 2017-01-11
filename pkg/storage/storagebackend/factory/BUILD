package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
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
        "//vendor:github.com/coreos/etcd/client",
        "//vendor:github.com/coreos/etcd/clientv3",
        "//vendor:github.com/coreos/etcd/pkg/transport",
        "//vendor:golang.org/x/net/context",
        "//vendor:k8s.io/apimachinery/pkg/util/net",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["tls_test.go"],
    library = ":go_default_library",
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
