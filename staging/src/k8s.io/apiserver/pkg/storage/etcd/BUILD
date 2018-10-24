package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = ["api_object_versioner_test.go"],
    embed = [":go_default_library"],
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/storage:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/storage/testing:go_default_library",
    ],
)

go_library(
    name = "go_default_library",
    srcs = [
        "api_object_versioner.go",
        "doc.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/apiserver/pkg/storage/etcd",
    importpath = "k8s.io/apiserver/pkg/storage/etcd",
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/api/meta:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/validation/field:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/storage:go_default_library",
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
