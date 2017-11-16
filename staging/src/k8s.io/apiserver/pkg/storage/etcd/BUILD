package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = [
        "api_object_versioner_test.go",
        "etcd_helper_test.go",
        "etcd_watcher_test.go",
    ],
    importpath = "k8s.io/apiserver/pkg/storage/etcd",
    library = ":go_default_library",
    deps = [
        "//vendor/github.com/coreos/etcd/client:go_default_library",
        "//vendor/golang.org/x/net/context:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/api/equality:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/api/testing:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/conversion:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/fields:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/labels:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/serializer:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/diff:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/watch:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/apis/example:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/apis/example/v1:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/storage:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/storage/etcd/etcdtest:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/storage/etcd/testing:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/storage/testing:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/storage/tests:go_default_library",
    ],
)

go_library(
    name = "go_default_library",
    srcs = [
        "api_object_versioner.go",
        "doc.go",
        "etcd_helper.go",
        "etcd_watcher.go",
    ],
    importpath = "k8s.io/apiserver/pkg/storage/etcd",
    deps = [
        "//vendor/github.com/coreos/etcd/client:go_default_library",
        "//vendor/github.com/golang/glog:go_default_library",
        "//vendor/golang.org/x/net/context:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/api/meta:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/conversion:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/cache:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/watch:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/storage:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/storage/etcd/metrics:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/storage/etcd/util:go_default_library",
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
        "//staging/src/k8s.io/apiserver/pkg/storage/etcd/etcdtest:all-srcs",
        "//staging/src/k8s.io/apiserver/pkg/storage/etcd/metrics:all-srcs",
        "//staging/src/k8s.io/apiserver/pkg/storage/etcd/testing:all-srcs",
        "//staging/src/k8s.io/apiserver/pkg/storage/etcd/util:all-srcs",
    ],
    tags = ["automanaged"],
)
