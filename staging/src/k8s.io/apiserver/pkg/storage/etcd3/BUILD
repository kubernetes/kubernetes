package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = [
        "compact_test.go",
        "store_test.go",
        "watcher_test.go",
    ],
    library = ":go_default_library",
    deps = [
        "//vendor/github.com/coreos/etcd/clientv3:go_default_library",
        "//vendor/github.com/coreos/etcd/etcdserver/api/v3rpc/rpctypes:go_default_library",
        "//vendor/github.com/coreos/etcd/integration:go_default_library",
        "//vendor/golang.org/x/net/context:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/api/testing:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/fields:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/labels:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/serializer:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/diff:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/wait:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/watch:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/apis/example:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/apis/example/v1:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/storage:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/storage/tests:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/storage/value:go_default_library",
    ],
)

go_library(
    name = "go_default_library",
    srcs = [
        "compact.go",
        "event.go",
        "store.go",
        "watcher.go",
    ],
    deps = [
        "//vendor/github.com/coreos/etcd/clientv3:go_default_library",
        "//vendor/github.com/coreos/etcd/etcdserver/api/v3rpc/rpctypes:go_default_library",
        "//vendor/github.com/coreos/etcd/mvcc/mvccpb:go_default_library",
        "//vendor/github.com/golang/glog:go_default_library",
        "//vendor/golang.org/x/net/context:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/api/meta:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/conversion:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/watch:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/storage:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/storage/etcd:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/storage/value:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/util/trace:go_default_library",
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
        "//staging/src/k8s.io/apiserver/pkg/storage/etcd3/preflight:all-srcs",
    ],
    tags = ["automanaged"],
)
