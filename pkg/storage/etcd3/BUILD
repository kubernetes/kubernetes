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
        "compact.go",
        "event.go",
        "store.go",
        "watcher.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/storage/etcd:go_default_library",
        "//vendor:github.com/coreos/etcd/clientv3",
        "//vendor:github.com/coreos/etcd/etcdserver/api/v3rpc/rpctypes",
        "//vendor:github.com/coreos/etcd/mvcc/mvccpb",
        "//vendor:github.com/golang/glog",
        "//vendor:golang.org/x/net/context",
        "//vendor:k8s.io/apimachinery/pkg/api/meta",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/conversion",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/watch",
        "//vendor:k8s.io/apiserver/pkg/storage",
        "//vendor:k8s.io/apiserver/pkg/util/trace",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "compact_test.go",
        "store_test.go",
        "watcher_test.go",
    ],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/coreos/etcd/clientv3",
        "//vendor:github.com/coreos/etcd/etcdserver/api/v3rpc/rpctypes",
        "//vendor:github.com/coreos/etcd/integration",
        "//vendor:golang.org/x/net/context",
        "//vendor:k8s.io/apimachinery/pkg/api/testing",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/fields",
        "//vendor:k8s.io/apimachinery/pkg/labels",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/runtime/serializer",
        "//vendor:k8s.io/apimachinery/pkg/util/wait",
        "//vendor:k8s.io/apimachinery/pkg/watch",
        "//vendor:k8s.io/apiserver/pkg/apis/example",
        "//vendor:k8s.io/apiserver/pkg/apis/example/v1",
        "//vendor:k8s.io/apiserver/pkg/storage",
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
