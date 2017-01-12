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
        "api_object_versioner.go",
        "doc.go",
        "etcd_helper.go",
        "etcd_watcher.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/storage:go_default_library",
        "//pkg/storage/etcd/util:go_default_library",
        "//pkg/util:go_default_library",
        "//vendor:github.com/coreos/etcd/client",
        "//vendor:github.com/golang/glog",
        "//vendor:golang.org/x/net/context",
        "//vendor:k8s.io/apimachinery/pkg/api/meta",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/conversion",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/util/runtime",
        "//vendor:k8s.io/apimachinery/pkg/watch",
        "//vendor:k8s.io/apiserver/pkg/storage/etcd/metrics",
        "//vendor:k8s.io/apiserver/pkg/util/cache",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "api_object_versioner_test.go",
        "etcd_helper_test.go",
        "etcd_watcher_test.go",
    ],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/testapi:go_default_library",
        "//pkg/api/testing:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//pkg/fields:go_default_library",
        "//pkg/storage:go_default_library",
        "//pkg/storage/etcd/etcdtest:go_default_library",
        "//pkg/storage/etcd/testing:go_default_library",
        "//pkg/storage/testing:go_default_library",
        "//vendor:github.com/coreos/etcd/client",
        "//vendor:golang.org/x/net/context",
        "//vendor:k8s.io/apimachinery/pkg/conversion",
        "//vendor:k8s.io/apimachinery/pkg/labels",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/serializer",
        "//vendor:k8s.io/apimachinery/pkg/watch",
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
        "//pkg/storage/etcd/etcdtest:all-srcs",
        "//pkg/storage/etcd/testing:all-srcs",
        "//pkg/storage/etcd/util:all-srcs",
    ],
    tags = ["automanaged"],
)
